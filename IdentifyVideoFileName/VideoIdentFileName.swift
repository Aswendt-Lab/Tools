// VideoIdentFileName.swift
// Recursive video renamer using Apple Vision OCR.
// Default output: <StudyID>_<Stage>_<Behavior>.mp4
// Example: SP_T1_11_2_P11_Gridwalk.mp4
//
// Main usage:
//   ./VideoIdentFileName "/path/to/Behavior"

import Foundation
import AVFoundation
import Vision
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// ====================== Config / CLI ======================

struct Config {
    let root: URL
    let behaviorOverride: String?
    let seconds: Double?
    let startOffset: Double
    let step: Double
    let retryStep: Double
    let overwrite: Bool
    let dryRun: Bool
    let includeNamed: Bool
    let maxP: Int
    let subjectPrefixes: [String]
    let csv: URL?
    let debugDir: URL?
    let debugLimit: Int
    let printConfig: Bool
    let fullFrameFallback: Bool
}

func normalizeDashes(_ s: String) -> String {
    s.replacingOccurrences(of: "\u{2013}", with: "-")
     .replacingOccurrences(of: "\u{2014}", with: "-")
     .replacingOccurrences(of: "\u{2212}", with: "-")
}

func parseArgs() -> Config? {
    let argv = CommandLine.arguments.map(normalizeDashes)
    var args = Array(argv.dropFirst())

    guard let rootPath = args.first else { return nil }
    args.removeFirst()

    func read(_ key: String, def: String? = nil) -> String? {
        if let i = args.firstIndex(of: key), i + 1 < args.count {
            let value = args[i + 1]
            args.removeSubrange(i...(i + 1))
            return value
        }
        return def
    }

    func flag(_ key: String) -> Bool {
        if let i = args.firstIndex(of: key) {
            args.remove(at: i)
            return true
        }
        return false
    }

    // Backward-compatible but optional:
    _ = read("--mode")

    let behaviorOverride = read("--behavior")
    let seconds = read("--seconds").flatMap(Double.init)
    let startOffset = Double(read("--start-offset", def: "0") ?? "0") ?? 0

    // Fast first pass; retry pass is slower only for failures.
    let step = Double(read("--step", def: "1.0") ?? "1.0") ?? 1.0
    let retryStep = Double(read("--retry-step", def: "0.25") ?? "0.25") ?? 0.25

    let overwrite = flag("--overwrite")
    let dryRun = flag("--dry-run")
    let includeNamed = flag("--include-named")
    let fullFrameFallback = !flag("--no-fullframe-fallback")

    let maxP = Int(read("--max-p", def: "60") ?? "60") ?? 60

    let prefixesRaw = read("--subject-prefixes", def: "GV,SP,SR,PB,CC") ?? "GV,SP,SR,PB,CC"
    let subjectPrefixes = prefixesRaw
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() }
        .filter { !$0.isEmpty }

    let csv = read("--csv").flatMap { URL(fileURLWithPath: $0) }
    let debugDir = read("--debug-dir").flatMap { URL(fileURLWithPath: $0) }
    let debugLimit = Int(read("--debug-limit", def: "20") ?? "20") ?? 20
    let printConfig = flag("--print-config")

    return Config(
        root: URL(fileURLWithPath: rootPath),
        behaviorOverride: behaviorOverride,
        seconds: seconds,
        startOffset: startOffset,
        step: step,
        retryStep: retryStep,
        overwrite: overwrite,
        dryRun: dryRun,
        includeNamed: includeNamed,
        maxP: maxP,
        subjectPrefixes: subjectPrefixes,
        csv: csv,
        debugDir: debugDir,
        debugLimit: debugLimit,
        printConfig: printConfig,
        fullFrameFallback: fullFrameFallback
    )
}

// ====================== Utilities ======================

let videoExts: Set<String> = ["mp4", "mov", "m4v", "avi", "mkv"]

func isVideo(_ url: URL) -> Bool {
    videoExts.contains(url.pathExtension.lowercased())
}

func enumeratedVideos(at root: URL) -> [URL] {
    let fm = FileManager.default
    var out: [URL] = []

    if let enumerator = fm.enumerator(
        at: root,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
    ) {
        for case let file as URL in enumerator {
            if isVideo(file) {
                out.append(file)
            }
        }
    }

    return out.sorted { $0.path < $1.path }
}

func inferBehavior(from url: URL, override: String?) -> String {
    if let override = override, !override.isEmpty {
        return override
    }

    let parts = url.pathComponents.map { $0.lowercased() }
    let joined = parts.joined(separator: "/")

    if joined.contains("gridwalk") || joined.contains("grid_walk") || joined.contains("grid-walk") {
        return "Gridwalk"
    }

    if joined.contains("cylinder") || parts.contains("cy") {
        return "Cylinder"
    }

    if joined.contains("rotatingbeam") ||
       joined.contains("rotating_beam") ||
       joined.contains("rotating-beam") ||
       joined.contains("rotating beam") ||
       parts.contains("rb") {
        return "RotatingBeam"
    }

    return "Unknown"
}

func uniqueBasename(in dir: URL, preferred name: String) -> String {
    let fm = FileManager.default
    let stem = (name as NSString).deletingPathExtension
    let ext = (name as NSString).pathExtension

    func exists(_ base: String) -> Bool {
        fm.fileExists(atPath: dir.appendingPathComponent(base).path)
    }

    if !exists(name) {
        return name
    }

    for i in 1...999 {
        let candidate = "\(stem)_" + String(format: "%02d", i) + (ext.isEmpty ? "" : ".\(ext)")
        if !exists(candidate) {
            return candidate
        }
    }

    return "\(stem)_\(UUID().uuidString.prefix(6))" + (ext.isEmpty ? "" : ".\(ext)")
}

func looksAlreadyNamed(_ url: URL) -> Bool {
    let name = url.deletingPathExtension().lastPathComponent.uppercased()

    let patterns = [
        #"^[A-Z]{2,4}_[A-Z]\d+_\d+_\d+_(P\d{1,2}|BASELINE)_[A-Z0-9]+(_\d{2})?$"#,
        #"^[A-Z]{2,4}_\d+_\d+_\d+_(P\d{1,2}|BASELINE)_[A-Z0-9]+(_\d{2})?$"#,
        #"^(P\d{1,2}|BASELINE)_[A-Z0-9]+(_\d{2})?$"#
    ]

    for pat in patterns {
        let re = try! NSRegularExpression(pattern: pat)
        let r = NSRange(location: 0, length: name.utf16.count)
        if re.firstMatch(in: name, options: [], range: r) != nil {
            return true
        }
    }

    return false
}

func appendCSV(_ url: URL, _ row: [String]) {
    let fm = FileManager.default

    if !fm.fileExists(atPath: url.path) {
        let header = "\"source\",\"stage\",\"subject\",\"behavior\",\"new_name\"\n"
        if let data = header.data(using: .utf8) {
            try? data.write(to: url)
        }
    }

    let line = row
        .map { $0.replacingOccurrences(of: "\"", with: "\"\"") }
        .map { "\"\($0)\"" }
        .joined(separator: ",") + "\n"

    if let handle = try? FileHandle(forWritingTo: url) {
        handle.seekToEndOfFile()
        if let data = line.data(using: .utf8) {
            handle.write(data)
        }
        handle.closeFile()
    }
}

func savePNG(_ cg: CGImage, to url: URL) {
    guard let dest = CGImageDestinationCreateWithURL(
        url as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else {
        return
    }

    CGImageDestinationAddImage(dest, cg, nil)
    CGImageDestinationFinalize(dest)
}

// ====================== AVFoundation ======================

func loadDurationSecondsSync(_ asset: AVAsset) -> Double {
    let sem = DispatchSemaphore(value: 0)
    var seconds: Double = 0

    Task {
        do {
            seconds = try await asset.load(.duration).seconds
        } catch {
            seconds = 0
        }
        sem.signal()
    }

    sem.wait()
    return seconds
}

func generateCGImageSync(_ gen: AVAssetImageGenerator, at time: CMTime) -> CGImage? {
    if #available(macOS 15.0, *) {
        let sem = DispatchSemaphore(value: 0)
        var out: CGImage?

        gen.generateCGImageAsynchronously(for: time) { cg, _, _ in
            out = cg
            sem.signal()
        }

        sem.wait()
        return out
    } else {
        var actual = CMTime.zero
        return try? gen.copyCGImage(at: time, actualTime: &actual)
    }
}

// ====================== Image helpers ======================

func upscaleCGImage(_ cg: CGImage, factor: Int = 4) -> CGImage {
    let factor = max(1, factor)
    let w = cg.width * factor
    let h = cg.height * factor

    guard let ctx = CGContext(
        data: nil,
        width: w,
        height: h,
        bitsPerComponent: 8,
        bytesPerRow: w * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        return cg
    }

    ctx.interpolationQuality = .high
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage() ?? cg
}

// ====================== Vision OCR ======================

func recognizeTexts(cg: CGImage, orientation: CGImagePropertyOrientation, prefixes: [String]) -> [(String, Float)] {
    let handler = VNImageRequestHandler(cgImage: cg, orientation: orientation, options: [:])
    let req = VNRecognizeTextRequest()

    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    req.recognitionLanguages = ["en-US"]

    let prefixWords = prefixes
    let tWords = (1...9).map { "T\($0)" }
    let pWords = (1...60).map { "P\($0)" }

    req.customWords = ["BASELINE"] + pWords + prefixWords + tWords

    var out: [(String, Float)] = []

    do {
        try handler.perform([req])
        if let observations = req.results {
            for obs in observations {
                for cand in obs.topCandidates(5) {
                    out.append((cand.string, Float(cand.confidence)))
                }
            }
        }
    } catch {
        // Ignore OCR failures for this frame/crop.
    }

    return out
}

func rotateOrientation(_ k: Int) -> CGImagePropertyOrientation {
    [.up, .right, .down, .left][k % 4]
}

// ====================== Parsing rules ======================

func normalizeStage(_ text: String, maxP: Int) -> String? {
    var t = text.uppercased()

    t = t.replacingOccurrences(of: "BA5ELINE", with: "BASELINE")
         .replacingOccurrences(of: "8ASELINE", with: "BASELINE")
         .replacingOccurrences(of: "BASELLNE", with: "BASELINE")
         .replacingOccurrences(of: "BASELlNE", with: "BASELINE")
         .replacingOccurrences(of: "O", with: "0")

    let noSep = t.replacingOccurrences(
        of: "[\\s_\\-]+",
        with: "",
        options: .regularExpression
    )

    if noSep.contains("BASELINE") {
        return "Baseline"
    }

    if let r = noSep.range(of: #"P(\d{1,2})"#, options: .regularExpression) {
        let nStr = noSep[r].dropFirst()
        if let n = Int(nStr), (1...maxP).contains(n) {
            return "P\(n)"
        }
    }

    return nil
}

func repairDigitPart(_ s: String) -> String {
    var out = ""

    for ch in s {
        switch ch {
        case "I", "L", "|": out.append("1")
        case "O", "Q": out.append("0")
        case "S": out.append("5")
        case "B": out.append("8")
        default: out.append(ch)
        }
    }

    return out
}

func normalizeSubject(_ text: String, prefixes: [String]) -> String? {
    var raw = text.uppercased()

    raw = raw.replacingOccurrences(of: "[:;,.]", with: "", options: .regularExpression)
             .replacingOccurrences(of: "[^A-Z0-9]+", with: "_", options: .regularExpression)
             .replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
             .trimmingCharacters(in: CharacterSet(charactersIn: "_"))

    let allowed = prefixes.isEmpty ? ["GV", "SP", "SR", "PB", "CC"] : prefixes

    guard let prefix = allowed.first(where: { raw.hasPrefix($0 + "_") || raw.hasPrefix($0) }) else {
        return nil
    }

    var rest = String(raw.dropFirst(prefix.count))
    rest = rest.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
    rest = repairDigitPart(rest)
    rest = rest.replacingOccurrences(of: "[^A-Z0-9]+", with: "_", options: .regularExpression)
               .replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
               .trimmingCharacters(in: CharacterSet(charactersIn: "_"))

    let candidate = prefix + "_" + rest

    let patterns = [
        #"^[A-Z]{2,4}_[A-Z]\d+_\d+_\d+$"#,
        #"^[A-Z]{2,4}_[A-Z]\d+_\d+$"#,
        #"^[A-Z]{2,4}_\d+_\d+_\d+$"#
    ]

    for pat in patterns {
        let re = try! NSRegularExpression(pattern: pat)
        let r = NSRange(location: 0, length: candidate.utf16.count)
        if re.firstMatch(in: candidate, options: [], range: r) != nil {
            return candidate
        }
    }

    let fallbackPattern = #"([A-Z]{2,4}_[A-Z]?\d+_\d+_\d+)"#
    let fallbackRe = try! NSRegularExpression(pattern: fallbackPattern)
    let r = NSRange(location: 0, length: candidate.utf16.count)

    if let m = fallbackRe.firstMatch(in: candidate, options: [], range: r) {
        return (candidate as NSString).substring(with: m.range)
    }

    return nil
}

// ====================== Yellow sticky detector ======================

struct Box {
    let x0: Int
    let y0: Int
    let x1: Int
    let y1: Int
}

func yellowCrops(from cg: CGImage) -> [CGImage] {
    let scale = max(1, max(cg.width, cg.height) / 640)
    let Wt = max(1, cg.width / scale)
    let Ht = max(1, cg.height / scale)

    guard let ctx = CGContext(
        data: nil,
        width: Wt,
        height: Ht,
        bitsPerComponent: 8,
        bytesPerRow: Wt * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        return []
    }

    ctx.interpolationQuality = .none
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: Wt, height: Ht))

    guard let buf = ctx.data?.assumingMemoryBound(to: UInt8.self) else {
        return []
    }

    var mask = [UInt8](repeating: 0, count: Wt * Ht)

    for y in 0..<Ht {
        for x in 0..<Wt {
            let p = (y * Wt + x) * 4
            let r = Int(buf[p])
            let g = Int(buf[p + 1])
            let b = Int(buf[p + 2])

            let c1 = (r > 170 && g > 170 && b < 195 && abs(r - g) < 65)
            let c2 = (r > 150 && g > 135 && b < 205 && r > g + 0)
            let c3 = (r > 130 && g > 120 && (r + g) > (b * 2 + 25))

            if c1 || c2 || c3 {
                mask[y * Wt + x] = 255
            }
        }
    }

    var visited = [Bool](repeating: false, count: Wt * Ht)
    var boxes: [Box] = []

    let minArea = max(40, (Wt * Ht) / 8000)
    let maxArea = Wt * Ht / 2
    let dirs = [(1,0), (-1,0), (0,1), (0,-1)]

    for y in 0..<Ht {
        for x in 0..<Wt {
            let idx = y * Wt + x

            if visited[idx] || mask[idx] == 0 {
                continue
            }

            var stack = [(x, y)]
            visited[idx] = true

            var minx = x
            var maxx = x
            var miny = y
            var maxy = y

            while let (cx, cy) = stack.popLast() {
                minx = min(minx, cx)
                maxx = max(maxx, cx)
                miny = min(miny, cy)
                maxy = max(maxy, cy)

                for (dx, dy) in dirs {
                    let nx = cx + dx
                    let ny = cy + dy

                    if nx >= 0 && ny >= 0 && nx < Wt && ny < Ht {
                        let nextIdx = ny * Wt + nx
                        if !visited[nextIdx] && mask[nextIdx] != 0 {
                            visited[nextIdx] = true
                            stack.append((nx, ny))
                        }
                    }
                }
            }

            let area = (maxx - minx + 1) * (maxy - miny + 1)

            if area < minArea || area > maxArea {
                continue
            }

            let width = maxx - minx + 1
            let height = maxy - miny + 1
            let ar = Double(max(width, 1)) / Double(max(height, 1))

            if ar < 0.25 || ar > 8.0 {
                continue
            }

            let pad = 10
            let x0 = max(0, (minx - pad) * scale)
            let y0 = max(0, (miny - pad) * scale)
            let x1 = min(cg.width, (maxx + pad + 1) * scale)
            let y1 = min(cg.height, (maxy + pad + 1) * scale)

            boxes.append(Box(x0: x0, y0: y0, x1: x1, y1: y1))
        }
    }

    // Merge overlapping boxes.
    var merged: [Box] = []

    func overlap(_ a: Box, _ b: Box) -> Bool {
        !(a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0)
    }

    for b in boxes.sorted(by: { $0.x0 == $1.x0 ? $0.y0 < $1.y0 : $0.x0 < $1.x0 }) {
        var added = false

        for i in 0..<merged.count {
            if overlap(merged[i], b) {
                let m = merged[i]
                merged[i] = Box(
                    x0: min(m.x0, b.x0),
                    y0: min(m.y0, b.y0),
                    x1: max(m.x1, b.x1),
                    y1: max(m.y1, b.y1)
                )
                added = true
                break
            }
        }

        if !added {
            merged.append(b)
        }
    }

    var crops: [CGImage] = []

    for m in merged {
        let rect = CGRect(
            x: m.x0,
            y: m.y0,
            width: max(1, m.x1 - m.x0),
            height: max(1, m.y1 - m.y0)
        )

        if let crop = cg.cropping(to: rect) {
            crops.append(crop)
        }
    }

    return crops
}

// ====================== Sampling ======================

func sampleTimes(duration: Double, start: Double, seconds: Double?, step: Double) -> [CMTime] {
    let t0 = max(0, start)
    let t1 = seconds == nil ? duration : min(duration, start + (seconds ?? duration))

    guard t1 > t0, step > 0 else {
        return []
    }

    var times: [CMTime] = []
    var t = t0

    while t < t1 {
        times.append(CMTime(seconds: t, preferredTimescale: 600))
        t += step
    }

    return times
}

func updateVotes(from cg: CGImage, cfg: Config, stageVotes: inout [String:Int], subjVotes: inout [String:Int]) {
    for k in 0..<4 {
        let texts = recognizeTexts(
            cg: cg,
            orientation: rotateOrientation(k),
            prefixes: cfg.subjectPrefixes
        )

        for (txt, _) in texts {
            if let st = normalizeStage(txt, maxP: cfg.maxP) {
                stageVotes[st, default: 0] += 1
            }

            if let sj = normalizeSubject(txt, prefixes: cfg.subjectPrefixes) {
                subjVotes[sj, default: 0] += 1
            }
        }
    }
}

// ====================== Per-video pipeline ======================

func processVideo(at url: URL, cfg: Config) {
    if !cfg.includeNamed && looksAlreadyNamed(url) {
        print("[SKIP] Already named: \(url.lastPathComponent)")
        return
    }

    let behavior = inferBehavior(from: url, override: cfg.behaviorOverride)

    let asset = AVURLAsset(url: url)
    let gen = AVAssetImageGenerator(asset: asset)

    gen.appliesPreferredTrackTransform = true
    gen.requestedTimeToleranceAfter = .zero
    gen.requestedTimeToleranceBefore = .zero

    let duration = loadDurationSecondsSync(asset)

    var stageVotes: [String:Int] = [:]
    var subjVotes: [String:Int] = [:]

    // Pass 1: fast yellow crop pass.
    let fastTimes = sampleTimes(
        duration: duration,
        start: cfg.startOffset,
        seconds: cfg.seconds,
        step: cfg.step
    )

    var debugSaved = 0

    for t in fastTimes {
        autoreleasepool {
            guard let cg0 = generateCGImageSync(gen, at: t) else {
                return
            }

            let crops = yellowCrops(from: cg0)

            if let dbg = cfg.debugDir, !crops.isEmpty, debugSaved < cfg.debugLimit {
                try? FileManager.default.createDirectory(at: dbg, withIntermediateDirectories: true)

                for (idx, c) in crops.enumerated() where debugSaved < cfg.debugLimit {
                    let name = "\(url.deletingPathExtension().lastPathComponent)_t\(Int(t.seconds * 1000))_c\(idx).png"
                    savePNG(c, to: dbg.appendingPathComponent(name))
                    debugSaved += 1
                }
            }

            for crop in crops {
                let bigCrop = upscaleCGImage(crop, factor: 4)
                updateVotes(from: bigCrop, cfg: cfg, stageVotes: &stageVotes, subjVotes: &subjVotes)
            }
        }

        if (stageVotes.values.max() ?? 0) >= 2 && (subjVotes.values.max() ?? 0) >= 2 {
            break
        }
    }

    // Pass 2: slower yellow crop retry if either stage or subject is missing.
    if stageVotes.isEmpty || subjVotes.isEmpty {
        let retryTimes = sampleTimes(
            duration: duration,
            start: cfg.startOffset,
            seconds: cfg.seconds,
            step: cfg.retryStep
        )

        for t in retryTimes {
            autoreleasepool {
                guard let cg0 = generateCGImageSync(gen, at: t) else {
                    return
                }

                let crops = yellowCrops(from: cg0)

                if let dbg = cfg.debugDir, !crops.isEmpty, debugSaved < cfg.debugLimit {
                    try? FileManager.default.createDirectory(at: dbg, withIntermediateDirectories: true)

                    for (idx, c) in crops.enumerated() where debugSaved < cfg.debugLimit {
                        let name = "\(url.deletingPathExtension().lastPathComponent)_retry_t\(Int(t.seconds * 1000))_c\(idx).png"
                        savePNG(c, to: dbg.appendingPathComponent(name))
                        debugSaved += 1
                    }
                }

                for crop in crops {
                    let bigCrop = upscaleCGImage(crop, factor: 4)
                    updateVotes(from: bigCrop, cfg: cfg, stageVotes: &stageVotes, subjVotes: &subjVotes)
                }
            }

            if (stageVotes.values.max() ?? 0) >= 2 && (subjVotes.values.max() ?? 0) >= 2 {
                break
            }
        }
    }

    // Pass 3: full-frame fallback, no upscaling, only if still needed.
    if cfg.fullFrameFallback && (stageVotes.isEmpty || subjVotes.isEmpty) {
        let fallbackTimes = sampleTimes(
            duration: duration,
            start: cfg.startOffset,
            seconds: cfg.seconds,
            step: max(0.5, cfg.step)
        )

        for t in fallbackTimes {
            autoreleasepool {
                guard let cg = generateCGImageSync(gen, at: t) else {
                    return
                }

                // No full-frame upscaling. It is memory-heavy and can cause "zsh: killed".
                updateVotes(from: cg, cfg: cfg, stageVotes: &stageVotes, subjVotes: &subjVotes)
            }

            if (stageVotes.values.max() ?? 0) >= 2 && (subjVotes.values.max() ?? 0) >= 2 {
                break
            }
        }
    }

    guard let bestStage = stageVotes.max(by: { $0.value < $1.value })?.key else {
        print("[SKIP] No valid stage: \(url.path)")
        return
    }

    let subject = subjVotes.max(by: { $0.value < $1.value })?.key
    let ext = url.pathExtension.isEmpty ? "" : "." + url.pathExtension.lowercased()

    let newBase: String
    if let subject = subject {
        newBase = "\(subject)_\(bestStage)_\(behavior)\(ext)"
    } else {
        newBase = "\(bestStage)_\(behavior)\(ext)"
    }

    let parent = url.deletingLastPathComponent()

    if url.lastPathComponent == newBase {
        print("[OK] Already named: \(url.lastPathComponent)")
        return
    }

    let finalName = cfg.overwrite ? newBase : uniqueBasename(in: parent, preferred: newBase)
    let dst = parent.appendingPathComponent(finalName)

    if cfg.dryRun {
        print("[DRY] \(url.lastPathComponent) -> \(dst.lastPathComponent)")
        return
    }

    do {
        try FileManager.default.copyItem(at: url, to: dst)
        print("[COPIED] \(url.lastPathComponent) -> \(dst.lastPathComponent)")
    } catch {
        print("[ERROR] copy: \(error.localizedDescription)")
    }

    if let csv = cfg.csv {
        appendCSV(csv, [url.lastPathComponent, bestStage, subject ?? "", behavior, finalName])
    }
}

// ====================== Main ======================

if let cfg = parseArgs() {
    if cfg.printConfig {
        let secs = cfg.seconds.map { String(format: "%.3f", $0) } ?? "nil"
        let prefixes = cfg.subjectPrefixes.joined(separator: ",")
        print("""
        Config:
          root=\(cfg.root.path)
          behaviorOverride=\(cfg.behaviorOverride ?? "nil")
          step=\(cfg.step)
          retryStep=\(cfg.retryStep)
          seconds=\(secs)
          startOffset=\(cfg.startOffset)
          overwrite=\(cfg.overwrite)
          dryRun=\(cfg.dryRun)
          includeNamed=\(cfg.includeNamed)
          fullFrameFallback=\(cfg.fullFrameFallback)
          prefixes=\(prefixes)
          csv=\(cfg.csv?.path ?? "nil")
          debugDir=\(cfg.debugDir?.path ?? "nil")
          debugLimit=\(cfg.debugLimit)
        """)
    }

    let videos = enumeratedVideos(at: cfg.root)

    if videos.isEmpty {
        print("No videos found.")
        exit(0)
    }

    print("Found \(videos.count) video(s). Processing...")

    for video in videos {
        processVideo(at: video, cfg: cfg)
    }
} else {
    print("""
    Usage:
      swift VideoIdentFileName.swift <root-folder> [options]

    Default behavior:
      Recursively scans <root-folder>, finds all video files, reads yellow sticky notes,
      infers the behavior from the folder path, and writes a corrected copy next to
      each original video.

    Examples:
      ./VideoIdentFileName "/Volumes/.../Behavior"

      ./VideoIdentFileName "/Volumes/.../Behavior" --dry-run

      ./VideoIdentFileName "/Volumes/.../Behavior" --step 1.0 --seconds 120

      ./VideoIdentFileName "/Volumes/.../Behavior/Gridwalk" --behavior Gridwalk --step 0.5

    Options:
      --behavior <name>
          Override inferred behavior. Otherwise inferred from path:
          Gridwalk, Cylinder, RotatingBeam.

      --seconds <N>
          Scan only the first N seconds of each video. Omit to scan full video.

      --start-offset <N>
          Start scanning after N seconds. Default: 0.

      --step <N>
          Fast first-pass sampling interval in seconds. Default: 1.0.

      --retry-step <N>
          Slower retry interval used only for videos where stage/subject is missing.
          Default: 0.25.

      --subject-prefixes "GV,SP,SR,PB,CC"
          Allowed study ID prefixes. Default: GV,SP,SR,PB,CC.

      --max-p <N>
          Accept P1..PN. Default: 60.

      --overwrite
          Use exact target filename even if it already exists.

      --dry-run
          Print planned copies without writing files.

      --include-named
          Also process files that already look correctly named.

      --no-fullframe-fallback
          Disable expensive full-frame OCR fallback.

      --csv <path>
          Write an audit CSV.

      --debug-dir <folder>
          Save yellow-note crops for inspection.

      --debug-limit <N>
          Max debug crops per video. Default: 20.

      --print-config
          Print parsed configuration before processing.
    """)
    exit(2)
}
