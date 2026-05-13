// VideoIdentFileName_v7.swift
// Rename videos by reading yellow sticky notes with Apple Vision OCR.
// Output pattern: <StudyID>_<Stage>_<Behavior>.mp4
// Example: SP_T1_11_2_P11_Gridwalk.mp4

import Foundation
import AVFoundation
import Vision
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// ====================== Config / CLI ======================

struct Config {
    let root: URL
    let mode: String
    let behavior: String
    let seconds: Double?
    let startOffset: Double
    let step: Double
    let outDir: URL?
    let overwrite: Bool
    let maxP: Int
    let subjectPrefixes: [String]
    let csv: URL?
    let debugDir: URL?
    let printConfig: Bool
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

    let mode = (read("--mode", def: "subject_stage") ?? "subject_stage").lowercased()
    let behavior = read("--behavior", def: "Gridwalk") ?? "Gridwalk"
    let seconds = read("--seconds").flatMap(Double.init)
    let startOffset = Double(read("--start-offset", def: "0") ?? "0") ?? 0
    let step = Double(read("--step", def: "0.5") ?? "0.5") ?? 0.5
    let outDir = read("--out-dir").flatMap { URL(fileURLWithPath: $0) }
    let overwrite = flag("--overwrite")
    let maxP = Int(read("--max-p", def: "60") ?? "60") ?? 60

    // Your project-specific default prefixes
    let prefixesRaw = read("--subject-prefixes", def: "GV,SP,SR,PB,CC") ?? "GV,SP,SR,PB,CC"
    let subjectPrefixes = prefixesRaw
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).uppercased() }
        .filter { !$0.isEmpty }

    let csv = read("--csv").flatMap { URL(fileURLWithPath: $0) }
    let debugDir = read("--debug-dir").flatMap { URL(fileURLWithPath: $0) }
    let printConfig = flag("--print-config")

    return Config(
        root: URL(fileURLWithPath: rootPath),
        mode: mode,
        behavior: behavior,
        seconds: seconds,
        startOffset: startOffset,
        step: step,
        outDir: outDir,
        overwrite: overwrite,
        maxP: maxP,
        subjectPrefixes: subjectPrefixes,
        csv: csv,
        debugDir: debugDir,
        printConfig: printConfig
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

    return out.sorted { $0.lastPathComponent < $1.lastPathComponent }
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

func appendCSV(_ url: URL, _ row: [String]) {
    let fm = FileManager.default

    if !fm.fileExists(atPath: url.path) {
        let header = "\"source\",\"stage\",\"subject\",\"new_name\"\n"
        try? header.data(using: .utf8)?.write(to: url)
    }

    let line = row
        .map { $0.replacingOccurrences(of: "\"", with: "\"\"") }
        .map { "\"\($0)\"" }
        .joined(separator: ",") + "\n"

    if let handle = try? FileHandle(forWritingTo: url) {
        handle.seekToEndOfFile()
        handle.write(line.data(using: .utf8)!)
        try? handle.close()
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
        #"^[A-Z]{2,4}_[A-Z]\d+_\d+_\d+$"#,  // SP_T1_11_2
        #"^[A-Z]{2,4}_[A-Z]\d+_\d+$"#,      // SP_T1_11
        #"^[A-Z]{2,4}_\d+_\d+_\d+$"#        // GV_1_11_2
    ]

    for pat in patterns {
        let re = try! NSRegularExpression(pattern: pat)
        let r = NSRange(location: 0, length: candidate.utf16.count)
        if re.firstMatch(in: candidate, options: [], range: r) != nil {
            return candidate
        }
    }

    // Last-resort extraction inside a noisy OCR string.
    let joined = candidate
    let fallbackPattern = #"([A-Z]{2,4}_[A-Z]?\d+_\d+_\d+)"#
    let fallbackRe = try! NSRegularExpression(pattern: fallbackPattern)
    let r = NSRange(location: 0, length: joined.utf16.count)

    if let m = fallbackRe.firstMatch(in: joined, options: [], range: r) {
        return (joined as NSString).substring(with: m.range)
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

    for b in boxes.sorted(by: { ($0.x0, $0.y0) < ($1.x0, $1.y0) }) {
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

// ====================== Per-video pipeline ======================

func processVideo(at url: URL, cfg: Config) {
    let asset = AVURLAsset(url: url)
    let gen = AVAssetImageGenerator(asset: asset)

    gen.appliesPreferredTrackTransform = true
    gen.requestedTimeToleranceAfter = .zero
    gen.requestedTimeToleranceBefore = .zero

    let duration = loadDurationSecondsSync(asset)
    let times = sampleTimes(
        duration: duration,
        start: cfg.startOffset,
        seconds: cfg.seconds,
        step: cfg.step
    )

    var stageVotes: [String:Int] = [:]
    var subjVotes: [String:Int] = [:]

    for t in times {
        guard let cg0 = generateCGImageSync(gen, at: t) else {
            continue
        }

        let crops = yellowCrops(from: cg0)

        if let dbg = cfg.debugDir, !crops.isEmpty {
            try? FileManager.default.createDirectory(at: dbg, withIntermediateDirectories: true)

            for (idx, c) in crops.enumerated() {
                let name = "\(url.deletingPathExtension().lastPathComponent)_t\(Int(t.seconds * 1000))_c\(idx).png"
                savePNG(c, to: dbg.appendingPathComponent(name))
            }
        }

        for crop in crops {
            let bigCrop = upscaleCGImage(crop, factor: 4)

            for k in 0..<4 {
                let texts = recognizeTexts(
                    cg: bigCrop,
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

        if (stageVotes.values.max() ?? 0) >= 2 && (subjVotes.values.max() ?? 0) >= 2 {
            break
        }
    }

    // Fallback: full-frame OCR if either stage or subject was missed.
    if stageVotes.isEmpty || subjVotes.isEmpty {
        let fallbackTimes = sampleTimes(
            duration: duration,
            start: cfg.startOffset,
            seconds: cfg.seconds,
            step: max(0.4, cfg.step / 2)
        )

        for t in fallbackTimes {
            guard let cg = generateCGImageSync(gen, at: t) else {
                continue
            }

            let bigFrame = upscaleCGImage(cg, factor: 2)

            for k in 0..<4 {
                let texts = recognizeTexts(
                    cg: bigFrame,
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

            if (stageVotes.values.max() ?? 0) >= 2 && (subjVotes.values.max() ?? 0) >= 2 {
                break
            }
        }
    }

    guard let bestStage = stageVotes.max(by: { $0.value < $1.value })?.key else {
        print("[SKIP] No valid stage in: \(url.path)")
        return
    }

    let subject = subjVotes.max(by: { $0.value < $1.value })?.key
    let ext = url.pathExtension.isEmpty ? "" : "." + url.pathExtension.lowercased()

    // Behavior is always appended.
    let newBase: String
    if let subject = subject {
        newBase = "\(subject)_\(bestStage)_\(cfg.behavior)\(ext)"
    } else {
        newBase = "\(bestStage)_\(cfg.behavior)\(ext)"
    }

    let fm = FileManager.default

    if let out = cfg.outDir {
        try? fm.createDirectory(at: out, withIntermediateDirectories: true)

        let finalName = cfg.overwrite ? newBase : uniqueBasename(in: out, preferred: newBase)
        let dst = out.appendingPathComponent(finalName)

        do {
            try fm.copyItem(at: url, to: dst)
            print("[COPIED] \(url.lastPathComponent) -> \(dst.lastPathComponent)")
        } catch {
            print("[ERROR] copy: \(error.localizedDescription)")
        }

        if let csv = cfg.csv {
            appendCSV(csv, [url.lastPathComponent, bestStage, subject ?? "", finalName])
        }
    } else {
        let parent = url.deletingLastPathComponent()

        if url.lastPathComponent == newBase {
            print("[OK] Already named: \(newBase)")
            return
        }

        let finalName = cfg.overwrite ? newBase : uniqueBasename(in: parent, preferred: newBase)
        let dst = parent.appendingPathComponent(finalName)

        do {
            try fm.moveItem(at: url, to: dst)
            print("[RENAMED] \(url.lastPathComponent) -> \(dst.lastPathComponent)")
        } catch {
            print("[ERROR] rename: \(error.localizedDescription)")
        }

        if let csv = cfg.csv {
            appendCSV(csv, [url.lastPathComponent, bestStage, subject ?? "", finalName])
        }
    }
}

// ====================== Main ======================

if let cfg = parseArgs() {
    if cfg.printConfig {
        let secs = cfg.seconds.map { String(format: "%.3f", $0) } ?? "nil"
        let prefixes = cfg.subjectPrefixes.joined(separator: ",")
        let outPath = cfg.outDir?.path ?? "(in-place)"

        print("""
        Config:
          mode=\(cfg.mode)
          behavior=\(cfg.behavior)
          step=\(cfg.step)
          seconds=\(secs)
          startOffset=\(cfg.startOffset)
          outDir=\(outPath)
          prefixes=\(prefixes)
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
      swift VideoIdentFileName_v7.swift <root> [options]

    Options:
      --mode <stage_behavior|subject_stage>   (default: subject_stage)
      --behavior <name>                       (default: Gridwalk)
      --seconds <N>                           (omit to scan whole clip)
      --start-offset <N>                      (default 0)
      --step <N>                              (default 0.5; smaller = slower, more thorough)
      --out-dir <folder>                      (copy there; omit to rename in place)
      --overwrite                             (allow overwrite instead of suffixing _01)
      --max-p <int>                           (default 60)
      --subject-prefixes "GV,SP,SR,PB,CC"     (default: GV,SP,SR,PB,CC)
      --csv <path>                            (write audit CSV)
      --debug-dir <folder>                    (save yellow crops)
      --print-config                          (print parsed config and continue)
    """)
    exit(2)
}