from mlir_gpu import FragmentCoord, Pipeline, fragment, fullscreen_vertex, sdf, shader_math as sm, vec3, vec4


RADIUS = 0.45


def capsule(point):
    xy = sm.rotate2(point.xy, 0.35)
    local = vec3(xy, point.z)
    local = local.with_components(y=local.y - sm.clamp(local.y, -0.5, 0.5))
    return sdf.sphere(local, RADIUS)


@fragment
def capsule_fragment(pixel: FragmentCoord):
    camera = (pixel - 64) / 64
    origin = vec3(0, 0, -3)
    ray = sm.normalize(vec3(camera.x, -camera.y, 1))
    distance = 0.0
    hit = False
    glow = vec3(0)

    for step in range(64):
        point = origin + ray * distance
        surface = capsule(point)
        if surface < 0.001:
            hit = True
            break
        distance += surface * 0.65
        glow += vec3(0.001, 0.002, 0.004)
        if distance > 8.0:
            break
        if surface > 0.2:
            continue
        glow += vec3(0.002, 0.005, 0.006)

    color = vec3(0.02, 0.04, 0.08)
    if hit:
        point = origin + ray * distance
        normal = sdf.normal(capsule, point)
        color = normal * 0.5 + 0.5
    return vec4(sm.clamp(color + glow, 0.0, 1.0), 1)


pipeline = Pipeline(fullscreen_vertex, capsule_fragment)
