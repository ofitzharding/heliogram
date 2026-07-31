// SmearTest — 120Hz display test patterns for the screen-camera channel experiment.
// Decides whether the mini-LED XDR panel (+ iPhone slo-mo camera) can carry
// 120Hz content alternation as a clean, high-contrast signal, or smears it.
//
// Build:  ./build.sh
// Run:    ./smeartest        (fullscreen on the main display)
//
// Keys:
//   1  alternate  — full-screen black/white flip every frame (60Hz square wave)
//   2  split      — top half alternates, bottom half static 50% gray
//                   (exposes backlight/local-dimming coupling into static regions)
//   3  bands      — 8 horizontal bands, phase-staggered alternation
//   4  checker    — full-screen checkerboard, inverted every frame
//   5  sweep      — single white band sweeping down, one panel-height per second
//   6  static     — static half black / half white (camera + panel reference)
//   ESC quits. HUD (drawn small, top-left) shows pattern + measured fps.
//
// The HUD is deliberately tiny so it does not pollute the captured field.

import Cocoa
import Metal
import MetalKit

// MARK: - Shaders (compiled at runtime, no .metallib needed)

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

struct VOut { float4 pos [[position]]; float3 color; };

struct Quad { float2 origin; float2 size; float3 color; };

vertex VOut quad_vert(uint vid [[vertex_id]],
                      uint iid [[instance_id]],
                      constant Quad *quads [[buffer(0)]]) {
    // two triangles per quad, positions in NDC
    float2 corners[6] = { float2(0,0), float2(1,0), float2(0,1),
                          float2(1,0), float2(1,1), float2(0,1) };
    Quad q = quads[iid];
    float2 p = q.origin + corners[vid] * q.size;   // in 0..1 screen space, y down
    VOut o;
    o.pos = float4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0, 1);
    o.color = q.color;
    return o;
}

fragment float4 quad_frag(VOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
"""

// MARK: - Pattern state

enum Pattern: Int {
    case alternate = 1, split, bands, checker, sweep, staticRef
    var name: String {
        switch self {
        case .alternate: return "1 alternate 120Hz"
        case .split:     return "2 split (top alt / bottom gray)"
        case .bands:     return "3 bands x8 staggered"
        case .checker:   return "4 checker invert"
        case .sweep:     return "5 sweep 1Hz"
        case .staticRef: return "6 static reference"
        }
    }
}

struct QuadData { var origin: SIMD2<Float>; var size: SIMD2<Float>; var color: SIMD3<Float> }

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pipeline: MTLRenderPipelineState
    var pattern: Pattern = .alternate
    var frameIndex: UInt64 = 0
    var startTime = CACurrentMediaTime()
    var lastReport = CACurrentMediaTime()
    var framesSinceReport = 0
    var measuredFPS: Double = 0
    weak var hud: NSTextField?

    // --auto: run this schedule with no keyboard input, then quit.
    // Order matters: the analyzer re-identifies each phase from its flicker
    // signature, so one continuous slo-mo clip covers the whole session.
    var autoMode = false
    let appStart = CACurrentMediaTime()
    let schedule: [(Pattern, Double)] = [(.alternate, 25), (.split, 25), (.staticRef, 25)]

    init(device: MTLDevice) {
        self.device = device
        self.queue = device.makeCommandQueue()!
        let lib = try! device.makeLibrary(source: shaderSource, options: nil)
        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = lib.makeFunction(name: "quad_vert")
        desc.fragmentFunction = lib.makeFunction(name: "quad_frag")
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        self.pipeline = try! device.makeRenderPipelineState(descriptor: desc)
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func quads(for pattern: Pattern, frame: UInt64, elapsed: Double) -> [QuadData] {
        let phase = frame % 2 == 0        // flips every frame
        let white = SIMD3<Float>(1, 1, 1)
        let black = SIMD3<Float>(0, 0, 0)
        let gray  = SIMD3<Float>(0.5, 0.5, 0.5)
        func full(_ c: SIMD3<Float>) -> QuadData {
            QuadData(origin: .init(0, 0), size: .init(1, 1), color: c)
        }
        switch pattern {
        case .alternate:
            return [full(phase ? white : black)]
        case .split:
            return [
                QuadData(origin: .init(0, 0),   size: .init(1, 0.5), color: phase ? white : black),
                QuadData(origin: .init(0, 0.5), size: .init(1, 0.5), color: gray),
            ]
        case .bands:
            var out: [QuadData] = []
            for i in 0..<8 {
                // band i flips with a per-band phase offset: band i is white when
                // (frame + i) is even. Adjacent bands are always in antiphase.
                let on = (frame + UInt64(i)) % 2 == 0
                out.append(QuadData(origin: .init(0, Float(i) / 8.0),
                                    size: .init(1, 1.0 / 8.0),
                                    color: on ? white : black))
            }
            return out
        case .checker:
            var out: [QuadData] = [full(phase ? white : black)]
            let n = 8
            for r in 0..<n {
                for c in 0..<n where (r + c) % 2 == 0 {
                    out.append(QuadData(origin: .init(Float(c) / Float(n), Float(r) / Float(n)),
                                        size: .init(1.0 / Float(n), 1.0 / Float(n)),
                                        color: phase ? black : white))
                }
            }
            return out
        case .sweep:
            let y = Float(elapsed.truncatingRemainder(dividingBy: 1.0)) // 1 sweep/sec
            return [full(black),
                    QuadData(origin: .init(0, y), size: .init(1, 0.05), color: white)]
        case .staticRef:
            return [
                QuadData(origin: .init(0, 0),   size: .init(1, 0.5), color: white),
                QuadData(origin: .init(0, 0.5), size: .init(1, 0.5), color: black),
            ]
        }
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor else { return }
        if autoMode {
            let e = CACurrentMediaTime() - appStart
            var t = 0.0
            var current: Pattern? = nil
            for (p, d) in schedule {
                if e < t + d { current = p; break }
                t += d
            }
            guard let p = current else {
                DispatchQueue.main.async { NSApp.terminate(nil) }
                return
            }
            pattern = p
        }
        let elapsed = CACurrentMediaTime() - startTime
        var qs = quads(for: pattern, frame: frameIndex, elapsed: elapsed)
        frameIndex += 1
        framesSinceReport += 1
        let now = CACurrentMediaTime()
        if now - lastReport >= 1.0 {
            measuredFPS = Double(framesSinceReport) / (now - lastReport)
            framesSinceReport = 0
            lastReport = now
            let fps = measuredFPS
            let name = pattern.name
            DispatchQueue.main.async { [weak self] in
                self?.hud?.stringValue = String(format: "%@  |  %.1f fps", name, fps)
                self?.hud?.textColor = fps > 100 ? .systemGreen : .systemRed
            }
        }

        let cmd = queue.makeCommandBuffer()!
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        let enc = cmd.makeRenderCommandEncoder(descriptor: rpd)!
        enc.setRenderPipelineState(pipeline)
        enc.setVertexBytes(&qs, length: MemoryLayout<QuadData>.stride * qs.count, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6, instanceCount: qs.count)
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }
}

// MARK: - App scaffolding

final class KeyView: MTKView {
    var renderer: Renderer?
    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 { NSApp.terminate(nil) } // ESC
        if let chars = event.characters, let n = Int(chars), let p = Pattern(rawValue: n) {
            renderer?.pattern = p
            renderer?.frameIndex = 0
            renderer?.startTime = CACurrentMediaTime()
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)

let screen = NSScreen.main!
let window = NSWindow(contentRect: screen.frame,
                      styleMask: [.borderless],
                      backing: .buffered, defer: false)
window.level = .mainMenu + 1
window.collectionBehavior = [.fullScreenPrimary]

let device = MTLCreateSystemDefaultDevice()!
let view = KeyView(frame: screen.frame, device: device)
view.colorPixelFormat = .bgra8Unorm
view.preferredFramesPerSecond = 120   // ProMotion: request the full 120
let renderer = Renderer(device: device)
renderer.autoMode = CommandLine.arguments.contains("--auto")
view.delegate = renderer
view.renderer = renderer

// Tiny HUD, top-left. Small enough to crop out of captures.
let hud = NSTextField(labelWithString: "starting…")
hud.frame = NSRect(x: 8, y: screen.frame.height - 24, width: 500, height: 18)
hud.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
hud.textColor = .systemRed
hud.backgroundColor = .clear
view.addSubview(hud)
renderer.hud = hud

window.contentView = view
window.makeKeyAndOrderFront(nil)
window.makeFirstResponder(view)
app.activate(ignoringOtherApps: true)

// Hide the cursor while the test runs.
NSCursor.hide()

app.run()
