from mlir_gpu import (
    FrameParameters, FragmentCoord, Pipeline, fragment, fullscreen_vertex,
    maximum, sdf, select, shader_math as sm, vec3, vec4,
)


@fragment
def march_budget(pixel: FragmentCoord, frame: FrameParameters):
    uv = (pixel - frame.resolution * 0.5) / frame.resolution.y
    ray = sm.normalize(vec3(uv.x, -uv.y, 1))
    center = vec3(sm.sin(frame.time) * 0.15, 0, 0)
    radius = 0.65 + sm.sin(frame.time) * 0.08
    cap = 4 if pixel.x < frame.resolution.x * 0.5 else 64
    distance = 0.0
    used = 0
    hit = False
    escaped = False
    for step in range(cap):
        used += 1
        surface = sdf.sphere(vec3(0, 0, -3) + ray * distance - center, radius)
        if surface < 0.002:
            hit = True
            break
        distance += maximum(surface * 0.75, 0.002)
        if distance > 8.0:
            escaped = True
            break
    return vec4(float(used) / 64.0, select(hit, 1.0, 0.0), select(escaped, 1.0, 0.15), 1)


pipeline = Pipeline(fullscreen_vertex, march_budget)
