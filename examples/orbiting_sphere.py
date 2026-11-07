from mlir_gpu import FrameParameters, FragmentCoord, Pipeline, fragment, fullscreen_vertex, sdf, shader_math as sm, vec3, vec4


@fragment
def orbiting_sphere(pixel: FragmentCoord, frame: FrameParameters):
    camera = (pixel - frame.resolution * 0.5) / frame.resolution.y
    ray = sm.normalize(vec3(camera.x, -camera.y, 1))
    origin = vec3(0, 0, -3)
    center = vec3(sm.sin(frame.time) * 0.65, sm.cos(frame.time * 0.7) * 0.25, 0)
    distance = 0.0
    hit = False
    for step in range(64):
        point = origin + ray * distance
        surface = sdf.sphere(point - center, 0.65)
        if surface < 0.001:
            hit = True
            break
        distance += surface * 0.65
        if distance > 8.0:
            break
    color = vec3(0.02, 0.04, 0.08)
    if hit:
        normal = sm.normalize(origin + ray * distance - center)
        color = normal * 0.5 + 0.5
    return vec4(color, 1)


pipeline = Pipeline(fullscreen_vertex, orbiting_sphere)
