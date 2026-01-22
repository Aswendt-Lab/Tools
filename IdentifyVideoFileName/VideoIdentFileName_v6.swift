// VideoIdentFileName_v6.swift
// Rename videos by reading yellow sticky notes (Apple Vision OCR, Apple Silicon-friendly)

import Foundation
import AVFoundation
import Vision
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// ====================== Config / CLI ======================
struct Config {
    let root: URL
    let mode: String                 // "stage_behavior" | "subject_stage"
    let behavior: String
    let seconds: Double?             // nil -> scan full video
    let startOffset: Double
    let step: Double
    let outDir: URL?
    let overwrite: Bool
    let maxP: Int
    let subjectPrefixes: [String]    // whitelist; empty -> no whitelist
    let csv: URL?                    // write audit CSV
    let debugDir: URL?               // save OCR crops (PNG)
    let printConfig: Bool
}

// Map common Unicode dashes to ASCII '-' so pasted flags work
func normalizeDashes(_ s: String) -> String {
    s.replacingOccurrences(of: "\u{2013}", with: "-")  // en dash –
     .replacingOccurrences(of: "\u{2014}", with: "-")  // em dash —
     .replacingOccurrences(of: "\u{2212}", with: "-")  // minus −
}

func parseArgs() -> Config? {
    let argv = CommandLine.arguments.map(normalizeDashes)
    var args = Array(argv.dropFirst())
    guard let rootPath = args.first else { return nil }
    args.removeFirst()

    func read(_ k: String, def: String? = nil) -> String? {
        if let i = args.firstIndex(of: k), i+1 < args.count {
            let v = args[i+1]; args.removeSubrange(i...(i+1)); return v
        }
        return def
    }
    func flag(_ k: String) -> Bool {
        if let i = args.firstIndex(of: k) { args.remove(at: i); return true }
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
    let prefixesRaw = read("--subject-prefixes", def: "GV,KS,SS,SO") ?? ""
    let subjectPrefixes = prefixesRaw.split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespaces).uppercased() }
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
let videoExts: Set<String> = ["mp4","mov","m4v","avi","mkv"]
func isVideo(_ u: URL) -> Bool { videoExts.contains(u.pathExtension.lowercased()) }

func enumeratedVideos(at root: URL) -> [URL] {
    let fm = FileManager.default
    var out: [URL] = []
    if let e = fm.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles]) {
        for case let f as URL in e where isVideo(f) { out.append(f) }
    }
    return out.sorted { $0.lastPathComponent < $1.lastPathComponent }
}

// unique name if exists: "name.ext" -> "name_01.ext", ...
func uniqueBasename(in dir: URL, preferred name: String) -> String {
    let fm = FileManager.default
    let stem = (name as NSString).deletingPathExtension
    let ext  = (name as NSString).pathExtension
    func path(_ base: String) -> String { dir.appendingPathComponent(base).path }
    if !fm.fileExists(atPath: path(name)) { return name }
    for i in 1...999 {
        let candidate = "\(stem)_" + String(format: "%02d", i) + (ext.isEmpty ? "" : ".\(ext)")
        if !fm.fileExists(atPath: path(candidate)) { return candidate }
    }
    return "\(stem)_" + UUID().uuidString.prefix(6) + (ext.isEmpty ? "" : ".\(ext)")
}

func appendCSV(_ url: URL, _ row: [String]) {
    let fm = FileManager.default
    let line = row.map { $0.replacingOccurrences(of: "\"", with: "\"\"") }
                  .map { "\"\($0)\"" }.joined(separator: ",") + "\n"
    if !fm.fileExists(atPath: url.path) {
        let header = "\"source\",\"stage\",\"subject\",\"new_name\"\n"
        try? header.data(using: .utf8)!.write(to: url)
    }
    if let h = try? FileHandle(forWritingTo: url) {
        h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); try? h.close()
    }
}

func savePNG(_ cg: CGImage, to url: URL) {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dest, cg, nil)
    CGImageDestinationFinalize(dest)
}

// ====================== AVFoundation (modern) ======================
func loadDurationSecondsSync(_ asset: AVAsset) -> Double {
    let sem = DispatchSemaphore(value: 0)
    var seconds: Double = 0
    Task { do { seconds = try await asset.load(.duration).seconds } catch { seconds = 0 }; sem.signal() }
    sem.wait(); return seconds
}

func generateCGImageSync(_ gen: AVAssetImageGenerator, at time: CMTime) -> CGImage? {
    if #available(macOS 15.0, *) {
        let sem = DispatchSemaphore(value: 0)
        var out: CGImage?
        gen.generateCGImageAsynchronously(for: time) { cg, _, _ in out = cg; sem.signal() }
        sem.wait(); return out
    } else {
        var actual = CMTime.zero
        return try? gen.copyCGImage(at: time, actualTime: &actual)
    }
}

// ====================== Vision OCR ======================
func recognizeTexts(cg: CGImage, orientation: CGImagePropertyOrientation) -> [(String, Float)] {
    let handler = VNImageRequestHandler(cgImage: cg, orientation: orientation, options: [:])
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = true
    req.recognitionLanguages = ["en-US"]
    req.customWords = ["BASELINE"] + (1...60).map { "P\($0)" }  // bias

    var out: [(String, Float)] = []
    do {
        try handler.perform([req])
        if let obs = req.results {
            for r in obs {
                for cand in r.topCandidates(3) { out.append((cand.string, Float(cand.confidence))) }
            }
        }
    } catch { /* ignore */ }
    return out
}

func rotateOrientation(_ k: Int) -> CGImagePropertyOrientation {
    [.up, .right, .down, .left][k % 4]
}

// ====================== Parsing rules ======================
func clean(_ s: String) -> String {
    let up = s.uppercased()
    let a = up.replacingOccurrences(of: "\\s+", with: "_", options: .regularExpression)
    let b = a.replacingOccurrences(of: "[^A-Z0-9_]", with: "", options: .regularExpression)
    return b
}

// replace characters frequently mis-OCR’d
func mapOCRConfusions(_ s: String) -> String {
    var out = ""
    for ch in s {
        switch ch {
        case "I","L","|": out.append("1")
        case "O","Q":     out.append("0")
        case "S":         out.append("5")
        case "B":         out.append("8")
        default:          out.append(ch)
        }
    }
    return out
}

func normalizeStage(_ text: String, maxP: Int) -> String? {
    var t = text.uppercased()
    t = t.replacingOccurrences(of: "BA5ELINE", with: "BASELINE")
         .replacingOccurrences(of: "8ASELINE", with: "BASELINE")
         .replacingOccurrences(of: "BASELLNE", with: "BASELINE")
         .replacingOccurrences(of: "O", with: "0")
    let noSep = t.replacingOccurrences(of: "[\\s_\\-]+", with: "", options: .regularExpression)
    if noSep.contains("BASELINE") { return "Baseline" }
    let collapsed = noSep.replacingOccurrences(of: #"P[\s_\-]+"#, with: "P", options: .regularExpression)
    if let r = collapsed.range(of: #"P(\d{1,2})"#, options: .regularExpression) {
        let nStr = collapsed[r].dropFirst()
        if let n = Int(nStr), (1...maxP).contains(n) { return "P\(n)" }
    }
    return nil
}

func normalizeSubject(_ text: String, prefixes: [String]) -> String? {
    var t = mapOCRConfusions(text.uppercased())
    t = t.replacingOccurrences(of: "[:;,.]", with: "", options: .regularExpression)
         .replacingOccurrences(of: "[^A-Z0-9]+", with: "_", options: .regularExpression)
         .replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
         .trimmingCharacters(in: CharacterSet(charactersIn: "_"))

    if !prefixes.isEmpty {
        let ok = prefixes.contains { t.hasPrefix($0 + "_") || t.hasPrefix($0) }
        if !ok { return nil }
    }

    // strict pattern: e.g., GV_T3_13_1
    let re1 = try! NSRegularExpression(pattern: #"^[A-Z]{2,4}(_[A-Z]?\d+){2,}$"#)
    let r = NSRange(location: 0, length: t.utf16.count)
    if re1.firstMatch(in: t, options: [], range: r) != nil { return t }

    // fallback: prefix + 2+ groups
    let re2 = try! NSRegularExpression(pattern: #"([A-Z]{2,4})(?:_[A-Z]?\d+){2,}"#)
    if let m = re2.firstMatch(in: t, options: [], range: r) {
        var s = (t as NSString).substring(with: m.range)
        s = s.replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
        return s
    }
    return nil
}

// ====================== Yellow sticky detector ======================
struct Box { let x0:Int; let y0:Int; let x1:Int; let y1:Int }

func yellowCrops(from cg: CGImage) -> [CGImage] {
    // downscale for speed
    let scale = max(1, max(cg.width, cg.height) / 640)
    let Wt = max(1, cg.width / scale), Ht = max(1, cg.height / scale)
    guard let ctx = CGContext(data: nil, width: Wt, height: Ht, bitsPerComponent: 8,
                              bytesPerRow: Wt*4, space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return [] }
    ctx.interpolationQuality = .none
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: Wt, height: Ht))
    guard let buf = ctx.data?.assumingMemoryBound(to: UInt8.self) else { return [] }

    // multi-threshold yellow mask
    var mask = [UInt8](repeating: 0, count: Wt*Ht)
    for y in 0..<Ht {
        for x in 0..<Wt {
            let p = (y*Wt + x)*4
            let r = Int(buf[p]), g = Int(buf[p+1]), b = Int(buf[p+2])
            let c1 = (r > 170 && g > 170 && b < 190 && abs(r-g) < 60)
            let c2 = (r > 155 && g > 140 && b < 200 && r > g+5)
            let c3 = (r > 135 && g > 125 && (r+g) > (b*2 + 30))
            if c1 || c2 || c3 { mask[y*Wt+x] = 255 }
        }
    }

    // connected components
    var visited = [Bool](repeating: false, count: Wt*Ht)
    var boxes:[Box] = []
    let minArea = max(80, (Wt*Ht)/4000), maxArea = Wt*Ht/2
    let dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for y in 0..<Ht {
        for x in 0..<Wt {
            let idx = y*Wt+x
            if visited[idx] || mask[idx] == 0 { continue }
            var stack = [(x,y)]; visited[idx] = true
            var minx=x,maxx=x,miny=y,maxy=y
            while let (cx,cy) = stack.popLast() {
                if cx<minx {minx=cx}; if cx>maxx {maxx=cx}
                if cy<miny {miny=cy}; if cy>maxy {maxy=cy}
                for (dx,dy) in dirs {
                    let nx=cx+dx, ny=cy+dy
                    if nx>=0 && ny>=0 && nx<Wt && ny<Ht {
                        let i2 = ny*Wt+nx
                        if !visited[i2] && mask[i2] != 0 { visited[i2]=true; stack.append((nx,ny)) }
                    }
                }
            }
            let area = (maxx-minx+1)*(maxy-miny+1)
            if area < minArea || area > maxArea { continue }
            let ar = Double(max(maxx-minx+1,1))/Double(max(maxy-miny+1,1))
            if ar < 0.3 || ar > 6.0 { continue }
            let pad = 6
            let x0 = max(0, (minx - pad) * scale)
            let y0 = max(0, (miny - pad) * scale)
            let x1 = min(cg.width,  (maxx + pad + 1) * scale)
            let y1 = min(cg.height, (maxy + pad + 1) * scale)
            boxes.append(Box(x0:x0,y0:y0,x1:x1,y1:y1))
        }
    }

    // merge overlapping boxes
    var merged:[Box] = []
    func overlap(_ a:Box,_ b:Box)->Bool { !(a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) }
    for b in boxes.sorted(by:{ ($0.x0,$0.y0) < ($1.x0,$1.y0) }) {
        var added=false
        for i in 0..<merged.count {
            if overlap(merged[i], b) {
                let m = merged[i]
                merged[i] = Box(x0:min(m.x0,b.x0), y0:min(m.y0,b.y0), x1:max(m.x1,b.x1), y1:max(m.y1,b.y1))
                added=true; break
            }
        }
        if !added { merged.append(b) }
    }

    // produce crops
    var crops:[CGImage] = []
    for m in merged {
        let rect = CGRect(x:m.x0,y:m.y0,width:m.x1-m.x0,height:m.y1-m.y0)
        if let c = cg.cropping(to: rect) { crops.append(c) }
    }
    return crops
}

// ====================== Sampling ======================
func sampleTimes(duration: Double, start: Double, seconds: Double?, step: Double) -> [CMTime] {
    let t0 = max(0, start)
    let t1 = seconds == nil ? duration : min(duration, start + (seconds ?? duration))
    guard t1 > t0, step > 0 else { return [] }
    var ts: [CMTime] = []; var t = t0
    while t < t1 { ts.append(CMTime(seconds: t, preferredTimescale: 600)); t += step }
    return ts
}

// ====================== Per-video pipeline ======================
func processVideo(at url: URL, cfg: Config) {
    let asset = AVURLAsset(url: url)
    let gen = AVAssetImageGenerator(asset: asset)
    gen.appliesPreferredTrackTransform = true
    gen.requestedTimeToleranceAfter  = .zero
    gen.requestedTimeToleranceBefore = .zero

    let duration = loadDurationSecondsSync(asset)
    let times = sampleTimes(duration: duration, start: cfg.startOffset, seconds: cfg.seconds, step: cfg.step)

    var stageVotes: [String:Int] = [:]
    var subjVotes:  [String:Int] = [:]

    // pass 1: yellow crops first (slow but focused)
    for t in times {
        guard let cg0 = generateCGImageSync(gen, at: t) else { continue }

        let crops = yellowCrops(from: cg0)
        if let dbg = cfg.debugDir, !crops.isEmpty {
            try? FileManager.default.createDirectory(at: dbg, withIntermediateDirectories: true)
            for (idx, c) in crops.enumerated() {
                let name = "\(url.deletingPathExtension().lastPathComponent)_t\(Int(t.seconds*1000))_c\(idx).png"
                savePNG(c, to: dbg.appendingPathComponent(name))
            }
        }

        for crop in crops {
            for k in 0..<4 {
                for (txt, _) in recognizeTexts(cg: crop, orientation: rotateOrientation(k)) {
                    if let st = normalizeStage(txt, maxP: cfg.maxP) { stageVotes[st, default: 0] += 1 }
                    if let sj = normalizeSubject(txt, prefixes: cfg.subjectPrefixes) { subjVotes[sj, default: 0] += 1 }
                }
            }
        }
        if (stageVotes.values.max() ?? 0) >= 2 && (subjVotes.values.max() ?? 0) >= 2 { break }
    }

    // pass 2: full-frame fallback if either missing
    if stageVotes.isEmpty || subjVotes.isEmpty {
        let times2 = sampleTimes(duration: duration, start: cfg.startOffset, seconds: cfg.seconds, step: max(0.4, cfg.step/2))
        for t in times2 {
            guard let cg = generateCGImageSync(gen, at: t) else { continue }
            for k in 0..<4 {
                for (txt, _) in recognizeTexts(cg: cg, orientation: rotateOrientation(k)) {
                    if let st = normalizeStage(txt, maxP: cfg.maxP) { stageVotes[st, default: 0] += 1 }
                    if let sj = normalizeSubject(txt, prefixes: cfg.subjectPrefixes) { subjVotes[sj, default: 0] += 1 }
                }
            }
            if (stageVotes.values.max() ?? 0) >= 2 && (subjVotes.values.max() ?? 0) >= 2 { break }
        }
    }

    guard let bestStage = stageVotes.max(by: { $0.value < $1.value })?.key else {
        print("[SKIP] No valid stage in: \(url.path)")
        return
    }

    let ext = url.pathExtension.isEmpty ? "" : "." + url.pathExtension.lowercased()
    let fm = FileManager.default

    // ALWAYS append behavior (both modes)
    let subject = subjVotes.max(by: { $0.value < $1.value })?.key
    let newBase: String
    if let s = subject {
        newBase = "\(s)_\(bestStage)_\(cfg.behavior)\(ext)"
    } else {
        newBase = "\(bestStage)_\(cfg.behavior)\(ext)"
    }

    // write (collision-safe). If renaming in place and name already exact, skip.
    if let out = cfg.outDir {
        try? fm.createDirectory(at: out, withIntermediateDirectories: true)
        let finalName = cfg.overwrite ? newBase : uniqueBasename(in: out, preferred: newBase)
        let dst = out.appendingPathComponent(finalName)
        do { try fm.copyItem(at: url, to: dst); print("[COPIED] \(url.lastPathComponent) -> \(dst.lastPathComponent)") }
        catch { print("[ERROR] copy: \(error.localizedDescription)") }
        if let csv = cfg.csv { appendCSV(csv, [url.lastPathComponent, bestStage, subject ?? "", finalName]) }
    } else {
        let parent = url.deletingLastPathComponent()
        if url.lastPathComponent == newBase { print("[OK] Already named: \(newBase)"); return }
        let finalName = cfg.overwrite ? newBase : uniqueBasename(in: parent, preferred: newBase)
        let dst = parent.appendingPathComponent(finalName)
        do { try fm.moveItem(at: url, to: dst); print("[RENAMED] \(url.lastPathComponent) -> \(dst.lastPathComponent)") }
        catch { print("[ERROR] rename: \(error.localizedDescription)") }
        if let csv = cfg.csv { appendCSV(csv, [url.lastPathComponent, bestStage, subject ?? "", finalName]) }
    }
}

// ====================== Main ======================
if let cfg = parseArgs() {
    if cfg.printConfig {
        let secs = cfg.seconds.map { String(format: "%.3f", $0) } ?? "nil"  // <-- fixed ambiguity
        let prefixes = cfg.subjectPrefixes.joined(separator: ",")
        let outPath = cfg.outDir?.path ?? "(in-place)"
        print("Config:\n  mode=\(cfg.mode)\n  behavior=\(cfg.behavior)\n  step=\(cfg.step)\n  seconds=\(secs)\n  startOffset=\(cfg.startOffset)\n  outDir=\(outPath)\n  prefixes=\(prefixes)")
    }
    let vids = enumeratedVideos(at: cfg.root)
    if vids.isEmpty { print("No videos found."); exit(0) }
    print("Found \(vids.count) video(s). Processing...")
    for v in vids { processVideo(at: v, cfg: cfg) }
} else {
    print("""
Usage:
  swift VideoIdentFileName_v6.swift <root> [options]

Options:
  --mode <stage_behavior|subject_stage>   (default: subject_stage)
  --behavior <name>                       (default: Gridwalk)
  --seconds <N>                           (omit to scan whole clip)
  --start-offset <N>                      (default 0)
  --step <N>                              (default 0.5; smaller = slower, more thorough)
  --out-dir <folder>                      (copy there; omit to rename in place)
  --overwrite                             (allow overwrite instead of suffixing _01)
  --max-p <int>                           (default 60)
  --subject-prefixes "GV,KS,SS,SO"        (prefix whitelist; empty=no whitelist)
  --csv <path>                            (write audit CSV)
  --debug-dir <folder>                    (save yellow crops)
  --print-config                          (print parsed config and continue)
""")
    exit(2)
}
