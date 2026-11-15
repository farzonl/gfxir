from mlir_gpu import (
    FrameParameters, FragmentCoord, Pipeline, fragment, fullscreen_vertex,
    maximum, sdf, shader_math as sm, vec3, vec4,
)


@fragment
def lit_sphere(pixel: FragmentCoord, frame: FrameParameters):
    uv = (pixel - frame.resolution * 0.5) / frame.resolution.y
    ray = sm.normalize(vec3(uv.x, -uv.y, 1))
    origin = vec3(0, 0, -3)
    distance = 0.0
    hit = False
    for step in range(64):
        surface = sdf.sphere(origin + ray * distance, 0.7)
        if surface < 0.002:
            hit = True
            break
        distance += maximum(surface * 0.75, 0.002)
        if distance > 8.0:
            break
    background = vec3(0.015, 0.025, 0.045)
    color = background
    if hit:
        point = origin + ray * distance
        normal = sdf.normal(lambda sample: sdf.sphere(sample, 0.7), point)
        light = sm.normalize(vec3(sm.cos(frame.time), 0.7, -sm.sin(frame.time)))
        diffuse = maximum(sm.dot(normal, light), 0.0)
        rim = (1.0 - maximum(sm.dot(normal, -ray), 0.0)) ** 3.0
        color = vec3(0.15, 0.5, 0.95) * (0.15 + diffuse * 0.85)
        color += vec3(0.3, 0.55, 1.0) * rim * 0.35
        color = sm.mix(background, color, sm.exp(-distance * 0.08))
    color = sm.pow(color / (color + 1.0), vec3(1.0 / 2.2))
    return vec4(color, 1)


pipeline = Pipeline(fullscreen_vertex, lit_sphere)
