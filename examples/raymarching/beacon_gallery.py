from mlir_gpu import (
    FrameParameters, FragmentCoord, Pipeline, fragment, fullscreen_vertex,
    maximum, minimum, sdf, shader_math as sm, vec3, vec4,
)

MAX_STEPS = 72
MAX_DISTANCE = 24.0


def gallery(point, time):
    cell_z = sdf.repeat(point.z, 4.0)
    height = sm.sin(point.z * 0.7 + time) * 0.15
    left = vec3(point.x + 1.15, point.y - height, cell_z)
    right = vec3(point.x - 1.15, point.y + height, cell_z)
    lights = minimum(sdf.sphere(left, 0.3), sdf.sphere(right, 0.3))
    pedestals = minimum(
        sdf.box(left + vec3(0, 1.3, 0), vec3(0.22, 0.7, 0.22)),
        sdf.box(right + vec3(0, 1.3, 0), vec3(0.22, 0.7, 0.22)),
    )
    room = minimum(2.0 - abs(point.x), 2.0 - abs(point.y))
    return vec3(minimum(room, minimum(lights, pedestals)), lights, pedestals)


@fragment
def beacon_gallery(pixel: FragmentCoord, frame: FrameParameters):
    uv = (pixel - frame.resolution * 0.5) / frame.resolution.y
    origin = vec3(sm.sin(frame.time * 0.35) * 0.3, 0.15, frame.time * 1.4)
    forward = sm.normalize(vec3(sm.sin(frame.time * 0.23) * 0.08, 0, 1))
    right = sm.normalize(sm.cross(vec3(0, 1, 0), forward))
    up = sm.cross(forward, right)
    ray = sm.normalize(forward * 0.85 + right * uv.x - up * uv.y)
    distance = 0.0
    glow = vec3(0)
    hit = False
    for step in range(MAX_STEPS):
        if distance > MAX_DISTANCE:
            break
        point = origin + ray * distance
        surface = gallery(point, frame.time)
        if surface.x < 0.002:
            hit = True
            break
        advance = maximum(surface.x * 0.75, 0.002)
        emission = sm.mix(vec3(0.9, 0.35, 0.08), vec3(0.35, 0.15, 0.9),
                          sm.sin(point.z * 0.6 + frame.time) * 0.5 + 0.5)
        glow += emission * (advance * 0.022 / (0.025 + surface.y * surface.y * 2.0))
        distance += advance
    background = vec3(0.015, 0.025, 0.045)
    color = background
    if hit:
        point = origin + ray * distance
        normal = sdf.normal(lambda sample: gallery(sample, frame.time).x, point)
        diffuse = maximum(sm.dot(normal, sm.normalize(vec3(-0.5, 0.8, -0.3))), 0.0)
        surface_uv = sm.mix(point.xz, point.yz, abs(normal.x))
        lines = abs(sm.fract(surface_uv * 0.75) - 0.5)
        footprint = maximum(distance / frame.resolution.y, 0.0001)
        grid = 1.0 - sm.smoothstep(0.025 - footprint, 0.025 + footprint, minimum(lines.x, lines.y))
        color = vec3(0.045, 0.06, 0.085) * (0.3 + diffuse * 0.7) + vec3(0.1, 0.3, 0.5) * grid * 0.2
        light = 1.0 - sm.smoothstep(0.002, 0.025, gallery(point, frame.time).y)
        color = sm.mix(color, vec3(1.8, 0.9, 0.3), light)
        color = sm.mix(background, color, sm.exp(-distance * 0.06))
    color += glow * 0.65
    color = maximum(color, 0.0)
    color = sm.pow(color / (color + 1.0), vec3(1.0 / 2.2))
    return vec4(color, 1)


pipeline = Pipeline(fullscreen_vertex, beacon_gallery)
