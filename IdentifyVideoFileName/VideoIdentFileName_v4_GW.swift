import Foundation
import AVFoundation
import Vision
import CoreGraphics

// ---------------- Config / CLI ----------------
struct Config {
    let root: URL
    let mode: String           // "stage_behavior" | "subject_stage"
    let behavior: String
    let seconds: Double?       // nil -> scan full video
    let startOffset: Double
    let step: Double
    let outDir: URL?
    let overwrite: Bool
    let maxP: Int
}

func parseArgs() -> Config? {
    var args = Array(CommandLine.arguments.dropFirst())
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
    let mode = read("--mode", def: "stage_behavior")!.lowercased()
    let behavior = read("--behavior", def: "Gridwalk")!
    let seconds = read("--seconds").flatMap(Double.init)
    let startOffset = Double(read("--start-offset", def: "0")!) ?? 0
    let step = Double(read("--step", def: "0.5")!) ?? 0.5
    let outDir = read("--out-dir").flatMap { URL(fileURLWithPath: $0) }
    let overwrite = flag("--overwrite")
    let maxP = Int(read("--max-p", def: "60")!) ?? 60
    return Config(root: URL(fileURLWithPath: rootPath), mode: mode, behavior: behavior,
                  seconds: seconds, startOffset: startOffset, step: step,
                  outDir: outDir, overwrite: overwrite, maxP: maxP)
}

// ---------------- Utilities ----------------
let videoExts: Set<String> = ["mp4","mov","m4v","avi","mkv"]
func isVideo(_ u: URL) -> Bool { videoExts.contains(u.pathExtension.lowercased()) }

func enumeratedVideos(at root: URL) -> [URL] {
    let fm = FileManager.default
    var out: [URL] = []
    if let e = fm.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey],
                             options: [.skipsHiddenFiles]) {
        for case let f as URL in e where isVideo(f) { out.append(f) }
    }
    return out.sorted { $0.lastPathComponent < $1.lastPathComponent }
}

// unique name if exists: "name.ext" -> "name_01.ext", "name_02.ext", ...
func uniqueBasename(in dir: URL, preferred name: String) -> String {
    let fm = FileManager.default
    let stem = (name as NSString).deletingPathExtension
    let ext  = (name as NSString).pathExtension
    func path(_ base: String) -> String { dir.appendingPathComponent(base).path }
    if !fm.fileExists(atPath: path(name)) { return name }
    for i in 1...999 {
        let candidate = "\(stem)_" + String(format: "%02d", i) + "." + ext
        if !fm.fileExists(atPath: path(candidate)) { return candidate }
    }
    return "\(stem)_" + UUID().uuidString.prefix(6) + "." + ext
}

// Modern duration + frame extraction
func loadDurationSecondsSync(_ asset: AVAsset) -> Double {
    let sem = DispatchSemaphore(value: 0)
    var seconds: Double = 0
    Task {
        do { seconds = try await asset.load(.duration).seconds } catch { seconds = 0 }
        sem.signal()
    }
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

// ---------------- Vision OCR ----------------
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
            for r in obs { for cand in r.topCandidates(3) { out.append((cand.string, Float(cand.confidence))) } }
        }
    } catch { /* ignore */ }
    return out
}
func rotateOrientation(_ k: Int) -> CGImagePropertyOrientation { [.up, .right, .down, .left][k % 4] }

// ---------------- Parsing rules ----------------
func clean(_ s: String) -> String {
    let up = s.uppercased()
    let a = up.replacingOccurrences(of: "\\s+", with: "_", options: .regularExpression)
    let b = a.replacingOccurrences(of: "[^A-Z0-9_]", with: "", options: .regularExpression)
    return b
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

// robust subject parser: repairs I/l/|→1, O/Q→0, S→5, B→8; accepts spaces/underscores/hyphens
func normalizeSubject(_ text: String) -> String? {
    var t = text.uppercased()
    // repair common OCR confusions
    let map: [Character:Character] = ["I":"1","L":"1","|":"1","O":"0","Q":"0","S":"5","B":"8"]
    t = String(t.map { map[$0] ?? $0 })
    // standardize separators
    t = t.replacingOccurrences(of: "[:;,.]", with: "", options: .regularExpression)
         .replacingOccurrences(of: "[^A-Z0-9]+", with: "_", options: .regularExpression)
         .replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
         .trimmingCharacters(in: CharacterSet(charactersIn: "_"))

    // Try a few patterns, most specific first
    let patterns = [
        #"^([A-Z]{2,4})(?:_?[A-Z]\d+|_\d+){2,}$"#,                 // GV_T3_13_1
        #"^([A-Z]{2,4})_T\d+(?:_\d+){1,3}$"#,                       // GV_T3_13 or GV_T3_13_1
        #"^([A-Z]{2,4})(?:_\d+){2,}$"#                              // GV_13_1 (rare)
    ]
    for pat in patterns {
        let re = try! NSRegularExpression(pattern: pat)
        let range = NSRange(location: 0, length: t.utf16.count)
        if re.firstMatch(in: t, options: [], range: range) != nil { return t }
    }

    // Last resort: find a plausible prefix + 3+ groups
    let re2 = try! NSRegularExpression(pattern: #"([A-Z]{2,4})(?:_[A-Z]?\d+){2,}"#)
    let range2 = NSRange(location: 0, length: t.utf16.count)
    if let m = re2.firstMatch(in: t, options: [], range: range2) {
        var s = (t as NSString).substring(with: m.range)
        s = s.replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
        return s
    }
    return nil
}

// ---------------- Yellow sticky detector (robust) ----------------
struct Box { let x0:Int; let y0:Int; let x1:Int; let y1:Int }

func yellowCrops(from cg: CGImage) -> [CGImage] {
    // downscale
    let scale = max(1, max(cg.width, cg.height) / 640)
    let Wt = max(1, cg.width / scale), Ht = max(1, cg.height / scale)
    guard let ctx = CGContext(data: nil, width: Wt, height: Ht, bitsPerComponent: 8,
                              bytesPerRow: Wt*4, space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return [] }
    ctx.interpolationQuality = .none
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: Wt, height: Ht))
    guard let buf = ctx.data?.assumingMemoryBound(to: UInt8.self) else { return [] }

    var mask = [UInt8](repeating: 0, count: Wt*Ht)
    func setMask(_ i:Int){ mask[i] = 255 }
    for y in 0..<Ht {
        for x in 0..<Wt {
            let p = (y*Wt + x)*4
            let r = Int(buf[p]), g = Int(buf[p+1]), b = Int(buf[p+2])
            let cond1 = (r > 170 && g > 170 && b < 190 && abs(r-g) < 60)
            let cond2 = (r > 155 && g > 140 && b < 200 && r > g+5)
            let cond3 = (r > 135 && g > 125 && (r+g) > (b*2 + 30))
            if cond1 || cond2 || cond3 { setMask(y*Wt+x) }
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
            var q:[(Int,Int)] = [(x,y)]; visited[idx] = true
            var minx=x,maxx=x,miny=y,maxy=y, count=0
            while let (cx,cy) = q.popLast() {
                count += 1
                if cx<minx {minx=cx}; if cx>maxx {maxx=cx}
                if cy<miny {miny=cy}; if cy>maxy {maxy=cy}
                for (dx,dy) in dirs {
                    let nx=cx+dx, ny=cy+dy
                    if nx>=0 && ny>=0 && nx<Wt && ny<Ht {
                        let i2 = ny*Wt+nx
                        if !visited[i2] && mask[i2] != 0 { visited[i2]=true; q.append((nx,ny)) }
                    }
                }
            }
            let area = (maxx-minx+1)*(maxy-miny+1)
            if area < minArea || area > maxArea { continue }
            let ar = Double(max(maxx-minx+1,1)) / Double(max(maxy-miny+1,1))
            if ar < 0.3 || ar > 6.0 { continue }
            let pad = 6
            let x0 = max(0, (minx - pad) * scale)
            let y0 = max(0, (miny - pad) * scale)
            let x1 = min(cg.width,  (maxx + pad + 1) * scale)
            let y1 = min(cg.height, (maxy + pad + 1) * scale)
            boxes.append(Box(x0:x0,y0:y0,x1:x1,y1:y1))
        }
    }
    // simple merge
    var merged:[Box] = []
    func overlap(_ a:Box,_ b:Box)->Bool {
        !(a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0)
    }
    for b in boxes.sorted(by:{ ($0.x0,$0.y0) < ($1.x0,$1.y0) }) {
        var added=false
        for i in 0..<merged.count {
            if overlap(merged[i], b) {
                let m = merged[i]
                merged[i] = Box(x0:min(m.x0,b.x0),y0:min(m.y0,b.y0),x1:max(m.x1,b.x1),y1:max(m.y1,b.y1))
                added=true; break
            }
        }
        if !added { merged.append(b) }
    }
    // crops
    var crops:[CGImage] = []
    for m in merged {
        let rect = CGRect(x:m.x0,y:m.y0,width:m.x1-m.x0,height:m.y1-m.y0)
        if let c = cg.cropping(to: rect) { crops.append(c) }
    }
    return crops
}

// ---------------- Sampling ----------------
func sampleTimes(duration: Double, start: Double, seconds: Double?, step: Double) -> [CMTime] {
    let t0 = max(0, start)
    let t1 = seconds == nil ? duration : min(duration, start + (seconds ?? duration))
    guard t1 > t0, step > 0 else { return [] }
    var ts: [CMTime] = []; var t = t0
    while t < t1 { ts.append(CMTime(seconds: t, preferredTimescale: 600)); t += step }
    return ts
}

// ---------------- Per-video pipeline ----------------
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

    // pass 1: yellow crops (slow but focused)
    for t in times {
        guard let cg0 = generateCGImageSync(gen, at: t) else { continue }
        let crops = yellowCrops(from: cg0)
        for crop in crops {
            for k in 0..<4 {
                for (txt, _) in recognizeTexts(cg: crop, orientation: rotateOrientation(k)) {
                    if let st = normalizeStage(txt, maxP: cfg.maxP) { stageVotes[st, default: 0] += 1 }
                    if let sj = normalizeSubject(txt)               { subjVotes[sj, default: 0]  += 1 }
                }
            }
        }
        if (stageVotes.values.max() ?? 0) >= 2 && (subjVotes.values.max() ?? 0) >= 2 { break }
    }

    // pass 2: full-frame fallback if needed
    if stageVotes.isEmpty || subjVotes.isEmpty {
        let times2 = sampleTimes(duration: duration, start: cfg.startOffset,
                                 seconds: cfg.seconds, step: max(0.4, cfg.step/2))
        for t in times2 {
            guard let cg = generateCGImageSync(gen, at: t) else { continue }
            for k in 0..<4 {
                for (txt, _) in recognizeTexts(cg: cg, orientation: rotateOrientation(k)) {
                    if let st = normalizeStage(txt, maxP: cfg.maxP) { stageVotes[st, default: 0] += 1 }
                    if let sj = normalizeSubject(txt)               { subjVotes[sj, default: 0]  += 1 }
                }
            }
            if (stageVotes.values.max() ?? 0) >= 2 && (subjVotes.values.max() ?? 0) >= 2 { break }
        }
    }

    // decide names
    guard let bestStage = stageVotes.max(by: { $0.value < $1.value })?.key else {
        print("[SKIP] No valid stage in: \(url.path)"); return
    }
    let ext = "." + url.pathExtension.lowercased()
    var newBase: String
    if cfg.mode == "subject_stage" {
        if let bestSubj = subjVotes.max(by: { $0.value < $1.value })?.key {
            newBase = "\(bestSubj)_\(bestStage)\(ext)"
        } else {
            // fall back to stage only if subject missing
            newBase = "\(bestStage)\(ext)"
        }
    } else {
        // stage_behavior
        if let bestSubj = subjVotes.max(by: { $0.value < $1.value })?.key {
            // helpful: include subject in front if found anyway
            newBase = "\(bestSubj)_\(bestStage)_\(cfg.behavior)\(ext)"
        } else {
            newBase = "\(bestStage)_\(cfg.behavior)\(ext)"
        }
    }

    // write (with collision-safe name)
    let fm = FileManager.default
    if let out = cfg.outDir {
        try? fm.createDirectory(at: out, withIntermediateDirectories: true)
        let unique = uniqueBasename(in: out, preferred: newBase)
        let dst = out.appendingPathComponent(unique)
        do { try fm.copyItem(at: url, to: dst); print("[COPIED] \(url.lastPathComponent) -> \(dst.lastPathComponent)") }
        catch { print("[ERROR] copy: \(error.localizedDescription)") }
    } else {
        let parent = url.deletingLastPathComponent()
        let unique = uniqueBasename(in: parent, preferred: newBase)
        let dst = parent.appendingPathComponent(unique)
        do { try fm.moveItem(at: url, to: dst); print("[RENAMED] \(url.lastPathComponent) -> \(dst.lastPathComponent)") }
        catch { print("[ERROR] rename: \(error.localizedDescription)") }
    }
}

// ---------------- Top-level main ----------------
if let cfg = parseArgs() {
    let vids = enumeratedVideos(at: cfg.root)
    if vids.isEmpty { print("No videos found."); exit(0) }
    print("Found \(vids.count) video(s). Processing...")
    for v in vids { processVideo(at: v, cfg: cfg) }
} else {
    print("""
Usage:
  swift VideoIdentFileName_v4_StickyStrong.swift <root> [options]

Options:
  --mode <stage_behavior|subject_stage>   (default: stage_behavior)
  --behavior <name>                       (default: Gridwalk)
  --seconds <N>                           (omit to scan whole clip)
  --start-offset <N>                      (default 0)
  --step <N>                              (default 0.5; smaller = slower, more thorough)
  --out-dir <folder>                      (copy there; omit to rename in place)
  --overwrite                             (allow overwrite)
  --max-p <int>                           (default 60)
""")
    exit(2)
}
