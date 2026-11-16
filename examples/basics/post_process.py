"""Animated emissive hand-and-eye tunnel with an LDR/HDR split comparison.

Inspired by the user-supplied Infinite Hamsa reference by Noztol.
Run with PYTHONPATH=src python -m examples.basics.post_process --target metal.
"""

import argparse
from pathlib import Path
import tempfile

from mlir_gpu import (
    FrameParameters, Pipeline, Sampler, Target, Texture2D, Toolchain,
    compile_pipeline, fragment, fullscreen_uv_vertex, maximum, minimum,
    sample_texture, shader_math as sm, vec2, vec3, vec4,
)
from mlir_gpu.viewer import (
    load_program, render_offscreen, show_directx_window, show_metal_window,
    show_window, with_post_process, write_png,
)


def capsule(point, start, end, radius):
    offset = point - start
    axis = end - start
    along = sm.clamp(sm.dot(offset, axis) / sm.dot(axis, axis), 0.0, 1.0)
    return sm.length(offset - axis * along) - radius


def smooth_union(a, b, radius):
    weight = sm.clamp(0.5 + 0.5 * (b - a) / radius, 0.0, 1.0)
    return sm.mix(b, a, weight) - radius * weight * (1.0 - weight)


def hand_distance(point):
    p = vec2(abs(point.x), point.y - 0.1)
    center = capsule(p, vec2(0.0, -0.2), vec2(0.0, 0.55), 0.09)
    finger = capsule(p, vec2(0.2, -0.2), vec2(0.22, 0.45), 0.08)
    thumb_top = capsule(p, vec2(0.25, -0.15), vec2(0.55, -0.05), 0.06)
    thumb_bottom = capsule(p, vec2(0.55, -0.05), vec2(0.25, -0.45), 0.06)
    thumb = smooth_union(thumb_top, thumb_bottom, 0.12)
    palm = sm.length(p - vec2(0.0, -0.25)) - 0.35
    return smooth_union(smooth_union(smooth_union(center, finger, 0.04), thumb, 0.06), palm, 0.1)


def eye_distances(point, time):
    p = point + vec2(0.0, 0.15)
    lens = maximum(
        sm.length(p - vec2(0.0, 0.18)),
        sm.length(p + vec2(0.0, 0.18)),
    ) - 0.2842534
    closed = capsule(p, vec2(-0.22, 0.0), vec2(0.22, 0.0), 0.002)
    blink = sm.pow(maximum(sm.sin(time * 2.5), 0.0), 12.0)
    eye = sm.mix(lens, closed, blink)
    return vec2(eye, maximum(sm.length(p) - 0.07, eye))


def hash_value(seed):
    p = sm.fract(seed * 0.1031)
    p *= p + 33.33
    return sm.fract(p * (p + p))


def fractal_emission(point, age, identity, depth):
    phase = sm.fract(age * 0.25)
    envelope = sm.pow(maximum(sm.sin(phase * 3.14159265), 0.0), 0.3)
    cycle = sm.floor(age * 0.25) + identity * 13.0 + 3.0
    fold_limit = sm.mix(1.0, 2.0, hash_value(cycle))
    shift = vec2(
        sm.mix(0.6, 1.6, hash_value(cycle + 23.0)),
        sm.mix(0.4, 1.9, hash_value(cycle + 37.0)),
    )
    p = sm.rotate2(point * (3.5 - 2.0 * envelope), age * 0.15)
    color = vec3(0.0)
    for iteration in range(7):
        p = vec2(abs(p.x), abs(p.y))
        scale = 1.3 / sm.clamp(sm.dot(p, p), 0.1, fold_limit)
        p = p * scale - shift * envelope
        palette = 0.5 + 0.5 * sm.cos(vec3(0.0, 1.5, 3.0) + age * 0.4 + float(iteration) * 0.8)
        width = 0.015 * depth
        lines = width / (abs(p.x) + width) + width / (abs(p.y) + width)
        points = 0.02 * depth / (sm.length(p) + width)
        color += palette * (lines + points) * 1.2 * envelope
    flash = sm.exp(-80.0 * phase) + sm.exp(-80.0 * (1.0 - phase))
    core = minimum(flash * 0.02 * depth / (sm.length(point) + 0.01 * depth), 3.0)
    return color * envelope + vec3(1.0, 0.85, 0.6) * core


def make_scene(split=False):
    aspect_scale = 0.5 if split else 1.0

    @fragment
    def emissive_tunnel(uv, frame: FrameParameters):
        screen = vec2(
            (uv.x - 0.5) * frame.resolution.x / frame.resolution.y * aspect_scale,
            0.5 - uv.y,
        )
        fog = vec3(0.002, 0.001, 0.006)
        color = fog
        travel = frame.time * 1.5
        for ring in range(24, 0, -1):
            depth = float(ring) - sm.fract(travel)
            if depth >= 0.1:
                identity = sm.floor(travel) + float(ring)
                distance_fade = sm.exp(-0.12 * depth)
                near_fade = sm.smoothstep(0.0, 0.5, depth)
                p = screen * depth * 2.5
                twist = identity * 0.4 + frame.time * 0.15
                for hand in range(8):
                    angle = float(hand) * 0.78539816 + twist
                    local = sm.rotate2(p, -angle) - vec2(0.0, 2.2)
                    if sm.length(local - vec2(0.0, -0.2)) < 1.4:
                        hand_sdf = hand_distance(local)
                        eye_sdf = eye_distances(local, frame.time)
                        hand_id = identity * 10.0 + float(hand)
                        age = frame.time + hand_id * 1.618
                        emission = fractal_emission(local + vec2(0.0, 0.15), age, hand_id, depth)
                        footprint = depth * 2.5 / frame.resolution.y
                        hand_mask = 1.0 - sm.smoothstep(-footprint, footprint, hand_sdf)
                        eye_mask = 1.0 - sm.smoothstep(-footprint, footprint, eye_sdf.x)
                        thickness = maximum(depth * 0.004, footprint * 0.5)
                        hand_glow = thickness / (abs(hand_sdf) + thickness)
                        eye_glow = thickness / (abs(eye_sdf.x) + thickness)
                        pupil_glow = thickness / (abs(eye_sdf.y) + thickness)
                        hand_glow *= sm.exp(-20.0 * maximum(hand_sdf, 0.0))
                        eye_glow *= sm.exp(-20.0 * maximum(eye_sdf.x, 0.0))
                        pupil_glow *= sm.exp(-20.0 * maximum(eye_sdf.y, 0.0))
                        pulse = 1.0 + 0.2 * sm.sin(age * 4.0)
                        outlines = (
                            vec3(0.3, 0.6, 1.0) * hand_glow
                            + vec3(1.0, 0.2, 0.8) * eye_glow
                            + vec3(1.0, 0.9, 0.2) * pupil_glow
                        ) * pulse * 3.0
                        layer = emission * hand_mask * (1.0 - eye_mask) + outlines
                        layer = sm.mix(fog, layer, distance_fade)
                        alpha = sm.clamp(hand_mask + hand_glow + eye_glow + pupil_glow, 0.0, 1.0) * near_fade
                        color = sm.mix(color, layer, alpha)
        return vec4(maximum(color, 0.0), 1.0)

    return Pipeline(fullscreen_uv_vertex, emissive_tunnel)


def make_post(view="split", exposure=1.5):
    @fragment
    def tunnel_tonemap(uv, frame: FrameParameters, scene_color: Texture2D[0], scene_sampler: Sampler[1]):
        sample_uv = uv
        if view == "split":
            sample_uv = vec2(sm.fract(uv.x * 2.0), uv.y)
        color = sample_texture(scene_color, scene_sampler, sample_uv).xyz
        if view == "split":
            if uv.x < 0.5:
                color = sm.floor(sm.clamp(color, 0.0, 1.0) * 255.0 + 0.5) / 255.0
        peak = maximum(color.x, maximum(color.y, color.z))
        mapped = color * exposure / (1.0 + peak * exposure)
        display = sm.pow(maximum(mapped, 0.0), vec3(0.8))
        if view == "split":
            if abs(uv.x - 0.5) * frame.resolution.x < 1.0:
                display = vec3(0.25)
        return vec4(display, 1.0)

    return Pipeline(fullscreen_uv_vertex, tunnel_tonemap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("vulkan", "directx", "metal"), default="vulkan")
    parser.add_argument("--provider", choices=("nacelle",), help="Execute DirectX DXIL on the CPU using Nacelle.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="Save a still instead of opening an animated window.")
    mode.add_argument("--frames", type=int, default=0, help="Window frame limit; 0 runs until closed.")
    parser.add_argument("--view", choices=("split", "hdr", "ldr"), default="split")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--time", type=float, help="Freeze animation at this time; stills default to 2 seconds.")
    parser.add_argument("--exposure", type=float, default=1.5)
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.exposure <= 100.0:
        parser.error("--exposure must be finite and between 0 (exclusive) and 100.")
    if args.provider == "nacelle" and (args.target != "directx" or args.output is None or args.validation):
        parser.error("--provider nacelle requires --target directx and --output, without --validation.")
    tools = Toolchain()
    with tempfile.TemporaryDirectory(prefix="gfx-post-example-") as tmp:
        directory = Path(tmp)
        target = Target(args.target)
        scene = compile_pipeline(make_scene(args.view == "split"), target, directory / "scene", tools)
        post = compile_pipeline(make_post(args.view, args.exposure), target, directory / "post", tools)
        format = "rgba8_unorm" if args.view == "ldr" else "rgba32_sfloat"
        program = with_post_process(load_program(scene), load_program(post), format=format)
        print("Left: RGBA8 LDR clipping simulation. Right: floating-point HDR, then tonemapped."
              if args.view == "split" else f"View: {args.view.upper()}.")
        if args.output is not None:
            time = 2.0 if args.time is None else args.time
            provider_kwargs = {"provider": args.provider} if args.provider is not None else {}
            image = render_offscreen(
                program, args.width, args.height, validation=args.validation, time=time, **provider_kwargs,
            )
            write_png(args.output, image)
            print(f"Rendered emissive tunnel on {image.device}: {args.output}")
        else:
            show = {"metal": show_metal_window, "directx": show_directx_window, "vulkan": show_window}[args.target]
            show(program, args.width, args.height, frames=args.frames, validation=args.validation, time=args.time)


if __name__ == "__main__":
    main()
