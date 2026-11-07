"""Python port of the user-provided Light Cycle shader by Noztol."""

from mlir_gpu import (
    FrameParameters, FragmentCoord, Pipeline, fragment, fullscreen_vertex,
    maximum, minimum, sdf, shader_math as sm, vec2, vec3, vec4,
)

MAX_STEPS = 110
MAX_DIST = 80.0
SURF_DIST = 0.001

COL_BG = vec3(0.004, 0.007, 0.014)
COL_CYAN = vec3(0.0, 0.85, 1.0)
COL_ORANGE = vec3(1.0, 0.28, 0.0)
COL_BLUE = vec3(0.04, 0.18, 0.75)


def get_path(z):
    return vec2(
        sm.sin(z * 0.05) * 2.4 + sm.cos(z * 0.025),
        sm.cos(z * 0.04) * 2.0 + sm.sin(z * 0.02) * 0.8,
    )


def to_tunnel_space(point, time):
    shifted = point.xy - get_path(point.z)
    rotated = sm.rotate2(shifted, point.z * 0.02 + time * 0.05)
    return point.with_components(xy=rotated)


def map_scene(world, time):
    point = to_tunnel_space(world, time)
    q = vec3(sdf.fold_square(point.xy), point.z)
    cell = q.with_components(z=sdf.repeat(q.z, 4.0))
    walls = 4.0 - q.x
    recess_center = sdf.box(cell - vec3(4, 0, 0), vec3(0.5, 1.488, 1.488)) - 0.012
    recess_side = sdf.box(cell - vec3(4, 2.75, 0), vec3(0.5, 0.738, 1.488)) - 0.012
    architecture = maximum(walls, -minimum(recess_center, recess_side))
    rail1 = sdf.box2(q.xy - vec2(4, 1.5), vec2(0.05))
    rail2 = sdf.box2(q.xy - vec2(4, 2.0), vec2(0.05))
    rail3 = sdf.box2(q.xy - vec2(4, 3.5), vec2(0.05))
    rails = minimum(minimum(rail1, rail2), rail3)
    ring1 = sdf.box(vec3(cell.x - 4, cell.y - 2, abs(cell.z) - 1.5), vec3(0.05, 2, 0.05))
    ring2 = sdf.box(vec3(cell.x - 4, cell.y - 2, abs(cell.z) - 2.0), vec3(0.05, 2, 0.05))
    cyan = minimum(rails, minimum(ring1, ring2))
    orange = sdf.box2(q.xy - vec2(3.92), vec2(0.11))
    return vec3(minimum(architecture, minimum(cyan, orange)), cyan, orange)


def box_filtered_line(coord, scale, line_width, footprint):
    width = maximum(footprint * scale, 0.0001)
    position = coord * scale
    a = position + 0.5 * width
    b = position - 0.5 * width
    coverage = (
        sm.floor(a) + minimum(sm.fract(a) / line_width, 1.0)
        - sm.floor(b) - minimum(sm.fract(b) / line_width, 1.0)
    ) / (width / line_width)
    return sm.clamp(coverage, 0.0, 1.0)


def anti_aliased_grid(world, normal, ray, distance, pixel_spread, time):
    point = to_tunnel_space(world, time)
    q = vec3(sdf.fold_square(point.xy), point.z)
    local_normal = normal.with_components(xy=sm.rotate2(normal.xy, point.z * 0.02 + time * 0.05))
    n_abs = abs(local_normal)
    if abs(point.y) > abs(point.x):
        n_abs = n_abs.with_components(xy=n_abs.yx)
    axis_weight = 1.0 - sm.smoothstep(0.35, 0.65, n_abs)
    cos_theta = maximum(abs(sm.dot(normal, ray)), 0.15)
    footprint = distance * pixel_spread * 1.35 / cos_theta
    gx_minor = box_filtered_line(q.x + 0.25, 2.0, 0.035, footprint) * axis_weight.x
    gy_minor = box_filtered_line(q.y + 0.25, 2.0, 0.035, footprint) * axis_weight.y
    gz_minor = box_filtered_line(q.z + 0.25, 2.0, 0.035, footprint) * axis_weight.z
    minor = maximum(gx_minor, maximum(gy_minor, gz_minor))
    gy_major = box_filtered_line(q.y + 0.5, 1.0, 0.032, footprint) * axis_weight.y
    gz_major = box_filtered_line(q.z + 0.5, 1.0, 0.032, footprint) * axis_weight.z
    major = maximum(gy_major, gz_major)
    coverage = major * 0.85 * sm.exp(-0.0007 * distance * distance)
    coverage += minor * 0.45 * sm.exp(-0.0016 * distance * distance)
    pulse = sm.sin(point.z * 0.35 - time * 3.0) * 0.5 + 0.5
    coverage *= 0.45 + 0.55 * sm.pow(pulse, 3.0)
    return sm.clamp(coverage, 0.0, 1.0)


@fragment
def light_cycle(pixel: FragmentCoord, frame: FrameParameters):
    # Shadertoy uses bottom-left pixel centers; GfxIR uses top-left.
    coord = vec2(pixel.x, frame.resolution.y - pixel.y)
    min_res = minimum(frame.resolution.x, frame.resolution.y)
    uv = (coord - frame.resolution * 0.5) / min_res
    pixel_spread = 1.0 / min_res
    uv *= 1.0 + sm.dot(uv, uv) * 0.05
    camera_z = frame.time * 8.0
    origin = vec3(get_path(camera_z), camera_z)
    target_z = camera_z + 5.0
    target = vec3(get_path(target_z), target_z)
    forward = sm.normalize(target - origin)
    bank = -forward.x * 0.45 + sm.sin(frame.time * 0.3) * 0.1
    up = vec3(sm.sin(bank), sm.cos(bank), 0)
    right = sm.normalize(sm.cross(forward, up))
    up = sm.cross(right, forward)
    ray = sm.normalize(forward * 0.9 + right * uv.x + up * uv.y)
    distance = 0.0
    glow = vec3(0)
    min_cyan_cone = 1e5
    min_orange_cone = 1e5
    hit_z = 0.0
    for step in range(MAX_STEPS):
        point = origin + ray * distance
        result = map_scene(point, frame.time)
        surface = result.x
        cyan = result.y
        orange = result.z
        cone_radius = maximum(distance * pixel_spread * 1.25, 0.003)
        min_cyan_cone = minimum(min_cyan_cone, cyan / cone_radius)
        min_orange_cone = minimum(min_orange_cone, orange / cone_radius)
        advance = maximum(surface * 0.65, 0.002)
        glow += COL_CYAN * (advance * 0.022 / (0.015 + cyan * cyan * 3.5))
        packet = sm.pow(sm.sin(point.z * 0.25 + frame.time * 10.0) * 0.5 + 0.5, 2.5)
        orange_emit = sm.mix(COL_ORANGE * 0.45, COL_ORANGE * 2.2, packet)
        glow += orange_emit * (advance * 0.035 / (0.012 + orange * orange * 3.5))
        glow += COL_BLUE * (advance * 0.0035 / (0.2 + surface * surface))
        if surface < SURF_DIST or distance > MAX_DIST:
            hit_z = point.z
            break
        distance += advance
    color = COL_BG
    # Preserve the supplied shader's shading after budget exhaustion.
    if distance < MAX_DIST:
        point = origin + ray * distance
        normal = sdf.normal(lambda sample: map_scene(sample, frame.time).x, point, 0.002)
        view = -ray
        fresnel = sm.pow(sm.clamp(1.0 - maximum(sm.dot(normal, view), 0.0), 0.0, 1.0), 3.0)
        grid = anti_aliased_grid(point, normal, ray, distance, pixel_spread, frame.time)
        base_color = vec3(0.008, 0.012, 0.022)
        rim_color = sm.mix(COL_BLUE, COL_CYAN, 0.55) * fresnel * 0.9
        grid_color = COL_CYAN * grid * 1.5
        color = base_color + rim_color + grid_color
        cyan_coverage = 1.0 - sm.smoothstep(-0.5, 1.2, min_cyan_cone)
        color = sm.mix(color, COL_CYAN * 2.4 + vec3(0.75), cyan_coverage)
        orange_coverage = 1.0 - sm.smoothstep(-0.5, 1.2, min_orange_cone)
        packet = sm.pow(sm.sin(hit_z * 0.25 + frame.time * 10.0) * 0.5 + 0.5, 2.5)
        orange_surface = sm.mix(COL_ORANGE, COL_ORANGE * 2.8 + vec3(0.95, 0.65, 0.25), packet)
        color = sm.mix(color, orange_surface, orange_coverage)
        fog = 1.0 - sm.exp(-0.00075 * distance * distance)
        color = sm.mix(color, COL_BG, fog)
    color += glow * 0.8
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14)
    q = coord / frame.resolution
    vignette = sm.pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.15)
    color *= sm.mix(0.65, 1.0, vignette)
    color = sm.pow(maximum(color, 0.0), vec3(1.0 / 2.2))
    dither = sm.fract(sm.sin(sm.dot(coord, vec2(12.9898, 78.233))) * 43758.5453)
    color += (dither - 0.5) / 255.0
    return vec4(color, 1)


pipeline = Pipeline(fullscreen_vertex, light_cycle)
