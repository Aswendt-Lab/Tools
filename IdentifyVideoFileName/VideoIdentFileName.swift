// VideoIdentFileName.swift
// Recursive video renamer using Apple Vision OCR.
// Default output: <StudyID>_<Stage>_<Behavior>.mp4
// Updated: protects already-correct filenames before OCR, improves white/light handwritten
//          label OCR, accepts prefix-less IDs such as T3_73_7, and prepends a configurable
//          default prefix.
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
    let defaultPrefix: String
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

    let defaultPrefix = (read("--default-prefix", def: subjectPrefixes.first ?? "GV") ?? "GV")
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .uppercased()

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
        defaultPrefix: defaultPrefix,
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

func normalizedFilenameStem(_ url: URL) -> String {
    return url.deletingPathExtension().lastPathComponent
        .uppercased()
        .replacingOccurrences(of: "[^A-Z0-9]+", with: "_", options: .regularExpression)
        .replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
        .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
}

func filenameContainsBehavior(_ url: URL, behavior: String) -> Bool {
    let name = normalizedFilenameStem(url)
    let b = behavior.uppercased()
        .replacingOccurrences(of: "[^A-Z0-9]+", with: "_", options: .regularExpression)
        .trimmingCharacters(in: CharacterSet(charactersIn: "_"))

    guard !b.isEmpty, b != "UNKNOWN" else { return false }
    return name == b || name.contains("_\(b)") || name.contains("\(b)_")
}

func looksAlreadyNamed(_ url: URL) -> Bool {
    // Strict fallback check for the full required format:
    //   StudyID_Day_Test.mp4
    // Examples:
    //   GV_T3_129_Baseline_Cylinder.mp4
    //   GV_T3_12_1_P7_Gridwalk.mp4
    // Baseline_Cylinder_01.mp4 is NOT considered correctly named because it lacks StudyID.
    let name = normalizedFilenameStem(url)

    let patterns = [
        #"^[A-Z]{2,4}_T\d+_\d+_\d+_(P\d{1,2}|BASELINE)_[A-Z0-9]+(_\d+)?$"#,
        #"^[A-Z]{2,4}_T\d+_\d+_(P\d{1,2}|BASELINE)_[A-Z0-9]+(_\d+)?$"#
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

func isAlreadyCompleteFilename(_ url: URL, behavior: String, prefixes: [String], maxP: Int) -> Bool {
    // Conservative early skip used BEFORE OCR. If the filename already contains:
    //   - a valid StudyID,
    //   - a valid stage/day,
    //   - the behavior name,
    // then OCR must not be used to create a new copy. This protects files such as
    // GV_T3_12_3_Baseline_Cylinder 3.mp4 from being changed by a bad OCR read.
    return filenameSubject(url, prefixes: prefixes) != nil
        && filenameStage(url, maxP: maxP) != nil
        && filenameContainsBehavior(url, behavior: behavior)
}

func regexFirst(_ pattern: String, in text: String) -> String? {
    let re = try! NSRegularExpression(pattern: pattern)
    let r = NSRange(location: 0, length: text.utf16.count)
    guard let m = re.firstMatch(in: text, options: [], range: r) else { return nil }
    return (text as NSString).substring(with: m.range)
}

func filenameSubject(_ url: URL, prefixes: [String]) -> String? {
    // Extracts a StudyID from the current filename, if present.
    // This is treated as ground truth to avoid OCR changing GV_T3_12_1 into GV_T3_129 etc.
    // Spaces, hyphens and duplicate-copy suffixes are normalized before matching.
    let name = normalizedFilenameStem(url)
    let allowed = prefixes.isEmpty ? ["GV", "SP", "SR", "PB", "CC"] : prefixes
    let prefixGroup = allowed.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|")

    // Prefer the more specific 4-part subject before the 3-part subject.
    let patterns = [
        "(?:^|_)((?:\(prefixGroup))_T\\d+_\\d+_\\d+)(?=_|$)",
        "(?:^|_)((?:\(prefixGroup))_T\\d+_\\d+)(?=_|$)"
    ]

    for pat in patterns {
        let re = try! NSRegularExpression(pattern: pat)
        let r = NSRange(location: 0, length: name.utf16.count)
        if let m = re.firstMatch(in: name, options: [], range: r), m.numberOfRanges > 1 {
            return (name as NSString).substring(with: m.range(at: 1))
        }
    }

    return nil
}

func filenameStage(_ url: URL, maxP: Int) -> String? {
    let name = url.deletingPathExtension().lastPathComponent.uppercased()
    return normalizeStage(name, maxP: maxP)
}

func appendCSV(_ url: URL, _ row: [String]) {
    let fm = FileManager.default

    if !fm.fileExists(atPath: url.path) {
        let header = "\"source\",\"filename_subject\",\"ocr_subject\",\"final_subject\",\"stage\",\"behavior\",\"new_name\",\"action\"\n"
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
    // Disable language correction because study IDs such as T3_73_7 are not words.
    // Language correction can silently alter or discard handwritten IDs.
    req.usesLanguageCorrection = false
    req.recognitionLanguages = ["en-US"]
    req.minimumTextHeight = 0.008

    let prefixWords = prefixes
    let tWords = (1...9).map { "T\($0)" }
    let pWords = (1...60).map { "P\($0)" }
    let digitWords = (1...999).map { "\($0)" }

    req.customWords = ["BASELINE", "STUDY", "ID"] + pWords + prefixWords + tWords + digitWords

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

func normalizeSubject(_ text: String, prefixes: [String], defaultPrefix: String) -> String? {
    // OCR often returns labels as e.g.
    //   "Study ID: GV T3 129"
    //   "Animal GV_T3_12_1 Baseline"
    // The older logic only worked when the OCR string started directly with GV/SP/...,
    // so files such as Baseline_Cylinder.mp4 were skipped if the visible label
    // contained additional words before the StudyID. This version searches for the
    // StudyID anywhere inside the OCR string.
    var raw = text.uppercased()

    raw = raw.replacingOccurrences(of: "[:;,.]", with: " ", options: .regularExpression)
             .replacingOccurrences(of: "[^A-Z0-9]+", with: "_", options: .regularExpression)
             .replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
             .trimmingCharacters(in: CharacterSet(charactersIn: "_"))

    let allowed = prefixes.isEmpty ? ["GV", "SP", "SR", "PB", "CC"] : prefixes
    let prefixGroup = allowed.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|")

    // Search the normalized OCR line for a complete StudyID anywhere in the text.
    // Order matters: prefer the more specific 4-part ID before the 3-part ID.
    let anywherePatterns = [
        #"(?:^|_)("# + prefixGroup + #")_[A-Z]\d+_\d+_\d+(?=_|$)"#,
        #"(?:^|_)("# + prefixGroup + #")_[A-Z]\d+_\d+(?=_|$)"#,
        #"(?:^|_)("# + prefixGroup + #")_\d+_\d+_\d+(?=_|$)"#
    ]

    for pat in anywherePatterns {
        let re = try! NSRegularExpression(pattern: pat)
        let r = NSRange(location: 0, length: raw.utf16.count)
        if let m = re.firstMatch(in: raw, options: [], range: r) {
            var hit = (raw as NSString).substring(with: m.range)
            hit = hit.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
            return hit
        }
    }

    // Many white handwritten labels in the Cylinder videos only contain the study
    // core, for example "T3-73-7" or "T3_73_7", without the project prefix.
    // Accept that form and prepend the default prefix, usually GV.
    let prefixlessPatterns = [
        #"(?:^|_)(T\d+_\d+_\d+)(?=_|$)"#,
        #"(?:^|_)(T\d+_\d+)(?=_|$)"#
    ]

    for pat in prefixlessPatterns {
        let re = try! NSRegularExpression(pattern: pat)
        let r = NSRange(location: 0, length: raw.utf16.count)
        if let m = re.firstMatch(in: raw, options: [], range: r) {
            let ns = raw as NSString
            var hit: String
            if m.numberOfRanges > 1 {
                hit = ns.substring(with: m.range(at: 1))
            } else {
                hit = ns.substring(with: m.range)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
            }
            hit = repairDigitPart(hit)
            return "\(defaultPrefix)_\(hit)"
        }
    }

    // Fallback: if OCR merges the prefix with the next token, repair the digit-ish
    // part after the prefix and validate again. This preserves the previous behavior.
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
        #"^[A-Z]{2,4}_T\d+_\d+_\d+$"#,
        #"^[A-Z]{2,4}_T\d+_\d+$"#
    ]

    for pat in patterns {
        let re = try! NSRegularExpression(pattern: pat)
        let r = NSRange(location: 0, length: candidate.utf16.count)
        if re.firstMatch(in: candidate, options: [], range: r) != nil {
            return candidate
        }
    }

    return nil
}

// ====================== Yellow / white label detector ======================

struct Box {
    let x0: Int
    let y0: Int
    let x1: Int
    let y1: Int
}

func labelCrops(from cg: CGImage) -> [CGImage] {
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

            let maxRGB = max(r, max(g, b))
            let minRGB = min(r, min(g, b))

            // Yellow sticky notes.
            let yellow1 = (r > 170 && g > 170 && b < 195 && abs(r - g) < 65)
            let yellow2 = (r > 150 && g > 135 && b < 205 && r > g + 0)
            let yellow3 = (r > 130 && g > 120 && (r + g) > (b * 2 + 25))

            // White or very light labels/paper.
            // Keep this conservative to avoid selecting the whole background.
            let white1 = (r > 190 && g > 190 && b > 185 && (maxRGB - minRGB) < 45)
            let white2 = (r > 175 && g > 175 && b > 170 && (maxRGB - minRGB) < 30)

            if yellow1 || yellow2 || yellow3 || white1 || white2 {
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

func strategicCrops(from cg: CGImage) -> [CGImage] {
    // Fallback crops for labels that are white/light on a bright background,
    // where color segmentation may not find a separate component. These crops are
    // much cheaper and safer than upscaling the full frame, and they help when the
    // study label sits on a white sheet rather than a yellow sticky note.
    let w = cg.width
    let h = cg.height

    let rects = [
        // Broad crops.
        CGRect(x: 0, y: 0, width: w, height: h / 2),
        CGRect(x: 0, y: h / 2, width: w, height: h / 2),
        CGRect(x: 0, y: 0, width: w / 2, height: h),
        CGRect(x: w / 2, y: 0, width: w / 2, height: h),

        // Focused lower-left crops. In many Cylinder recordings the handwritten
        // white animal label is held at the lower-left edge of the frame.
        CGRect(x: 0, y: h * 45 / 100, width: w * 45 / 100, height: h * 45 / 100),
        CGRect(x: 0, y: h * 55 / 100, width: w * 50 / 100, height: h * 35 / 100),
        CGRect(x: 0, y: h * 60 / 100, width: w * 55 / 100, height: h * 30 / 100),
        CGRect(x: 0, y: h * 65 / 100, width: w * 60 / 100, height: h * 25 / 100),
        CGRect(x: 0, y: h * 50 / 100, width: w * 35 / 100, height: h * 35 / 100),

        // Focused upper-left for yellow Baseline/P-stage sticky notes.
        CGRect(x: 0, y: h * 20 / 100, width: w * 40 / 100, height: h * 45 / 100),

        CGRect(x: w / 10, y: h / 10, width: w * 8 / 10, height: h * 8 / 10),
        CGRect(x: 0, y: 0, width: w, height: h)
    ]

    var out: [CGImage] = []

    for rect in rects {
        if let crop = cg.cropping(to: rect) {
            out.append(crop)
        }
    }

    return out
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

            if let sj = normalizeSubject(txt, prefixes: cfg.subjectPrefixes, defaultPrefix: cfg.defaultPrefix) {
                subjVotes[sj, default: 0] += 1
            }
        }
    }
}

// ====================== Per-video pipeline ======================

func processVideo(at url: URL, cfg: Config) {
    let behavior = inferBehavior(from: url, override: cfg.behaviorOverride)
    let subjectFromFilename = filenameSubject(url, prefixes: cfg.subjectPrefixes)
    let stageFromFilename = filenameStage(url, maxP: cfg.maxP)

    if !cfg.includeNamed && isAlreadyCompleteFilename(url, behavior: behavior, prefixes: cfg.subjectPrefixes, maxP: cfg.maxP) {
        print("[SKIP] Already named: \(url.lastPathComponent)")
        return
    }

    if !cfg.includeNamed && looksAlreadyNamed(url) {
        print("[SKIP] Already named: \(url.lastPathComponent)")
        return
    }

    let asset = AVURLAsset(url: url)
    let gen = AVAssetImageGenerator(asset: asset)

    gen.appliesPreferredTrackTransform = true
    gen.requestedTimeToleranceAfter = .zero
    gen.requestedTimeToleranceBefore = .zero

    let duration = loadDurationSecondsSync(asset)

    var stageVotes: [String:Int] = [:]
    var subjVotes: [String:Int] = [:]

    // Pass 1: fast yellow/white label crop pass.
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

            let crops = labelCrops(from: cg0)

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

    // Pass 2: slower yellow/white label crop retry if either stage or subject is missing.
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

                let crops = labelCrops(from: cg0)

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

    // Pass 2b: strategic crops if color-based label detection missed a white/light label.
    if stageVotes.isEmpty || subjVotes.isEmpty {
        let cropFallbackTimes = sampleTimes(
            duration: duration,
            start: cfg.startOffset,
            seconds: cfg.seconds,
            step: max(1.0, cfg.step)
        )

        for t in cropFallbackTimes {
            autoreleasepool {
                guard let cg0 = generateCGImageSync(gen, at: t) else {
                    return
                }

                for crop in strategicCrops(from: cg0) {
                    // Moderate upscale: improves OCR on labels without the memory cost
                    // of full-frame 4x upscaling.
                    let bigCrop = upscaleCGImage(crop, factor: 3)
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

    let stageFromOCR = stageVotes.max(by: { $0.value < $1.value })?.key
    guard let bestStage = stageFromFilename ?? stageFromOCR else {
        print("[SKIP] No valid stage: \(url.path)")
        if let csv = cfg.csv {
            appendCSV(csv, [url.lastPathComponent, subjectFromFilename ?? "", "", "", "", behavior, "", "skip_no_stage"])
        }
        return
    }

    let subjectFromOCR = subjVotes.max(by: { $0.value < $1.value })?.key

    if let f = subjectFromFilename, let o = subjectFromOCR, f != o {
        print("[WARN] OCR subject conflict ignored: filename=\(f) detected=\(o) file=\(url.lastPathComponent)")
    }

    guard let finalSubject = subjectFromFilename ?? subjectFromOCR else {
        // This catches files such as Baseline_Cylinder_01.mp4 when the StudyID could not be read.
        print("[SKIP] No valid StudyID: \(url.path)")
        if let csv = cfg.csv {
            appendCSV(csv, [url.lastPathComponent, subjectFromFilename ?? "", subjectFromOCR ?? "", "", bestStage, behavior, "", "skip_no_study_id"])
        }
        return
    }

    let ext = url.pathExtension.isEmpty ? "" : "." + url.pathExtension.lowercased()
    let newBase = "\(finalSubject)_\(bestStage)_\(behavior)\(ext)"

    let parent = url.deletingLastPathComponent()

    if url.lastPathComponent == newBase {
        print("[OK] Already named: \(url.lastPathComponent)")
        return
    }

    let finalName = cfg.overwrite ? newBase : uniqueBasename(in: parent, preferred: newBase)
    let dst = parent.appendingPathComponent(finalName)

    if cfg.dryRun {
        print("[DRY] \(url.lastPathComponent) -> \(dst.lastPathComponent)")
        if let csv = cfg.csv {
            appendCSV(csv, [url.lastPathComponent, subjectFromFilename ?? "", subjectFromOCR ?? "", finalSubject, bestStage, behavior, finalName, "dry"])
        }
        return
    }

    do {
        try FileManager.default.copyItem(at: url, to: dst)
        print("[COPIED] \(url.lastPathComponent) -> \(dst.lastPathComponent)")
        if let csv = cfg.csv {
            appendCSV(csv, [url.lastPathComponent, subjectFromFilename ?? "", subjectFromOCR ?? "", finalSubject, bestStage, behavior, finalName, "copied"])
        }
    } catch {
        print("[ERROR] copy: \(error.localizedDescription)")
        if let csv = cfg.csv {
            appendCSV(csv, [url.lastPathComponent, subjectFromFilename ?? "", subjectFromOCR ?? "", finalSubject, bestStage, behavior, finalName, "error_copy"])
        }
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
          defaultPrefix=\(cfg.defaultPrefix)
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
      Recursively scans <root-folder>, finds all video files, reads yellow or white/light labels, accepts prefix-less handwritten IDs such as T3_73_7, uses OCR fallback crops,
      infers the behavior from the folder path, and writes a corrected copy next to
      each original video. Files are considered already named only if they follow:
      StudyID_Day_Test.mp4, e.g. GV_T3_129_Baseline_Cylinder.mp4.

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

      --default-prefix <prefix>
          Prefix used when the visible handwritten label contains only the core ID,
          e.g. T3_73_7. Default: first subject prefix, usually GV.

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
          Save yellow/white label crops for inspection.

      --debug-limit <N>
          Max debug crops per video. Default: 20.

      --print-config
          Print parsed configuration before processing.
    """)
    exit(2)
}
