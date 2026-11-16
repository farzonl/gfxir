"""Shared linear-HDR post processing for MiniGfx."""

import math

from mlir_gpu import Pipeline, Sampler, Texture2D, fragment, fullscreen_uv_vertex, maximum, sample_texture, select, vec4
from mlir_gpu import shader_math as sm
from mlir_gpu.parameters import struct
from mlir_gpu.frontend import Type


def validate_exposure(value):
    exposure = float(value)
    if not math.isfinite(exposure) or not 0.01 <= exposure <= 64.0:
        raise ValueError("Exposure must be finite and between 0.01 and 64")
    return exposure


def tonemap(color, exposure, comparison, uv):
    rgb = maximum(color.xyz, 0.0)
    rgb = rgb * exposure
    ldr = sm.clamp((rgb * (rgb * 2.51 + 0.03)) / (rgb * (rgb * 2.43 + 0.59) + 0.14), 0.0, 1.0)
    result = vec4(sm.pow(ldr, 1.0 / 2.2), color.w)
    if comparison:
        # Comparison scene shaders already produce display-referred LDR on the left.
        ldr_color = vec4(sm.floor(sm.clamp(color.xyz, 0.0, 1.0) * 255.0 + 0.5) / 255.0, color.w)
        return select(uv.x < 0.5, ldr_color, result)
    return result


def display_to_scene(color, exposure):
    """Invert ACES/gamma for display-referred overlays drawn into the HDR scene."""
    y = sm.pow(sm.clamp(color.xyz, 0.0, 1.0), 2.2)
    a = y * 2.43 - 2.51
    b = y * 0.59 - 0.03
    rgb = (-b - sm.sqrt(b * b - 4.0 * a * y * 0.14)) / (2.0 * a)
    return vec4(rgb / exposure, color.w)


@struct
class PostParams:
    exposure: Type.F32
    comparison: Type.F32
    padding: Type.VEC2


@fragment
def hdr_post(uv, scene_color: Texture2D[0], scene_sampler: Sampler[1], post: PostParams):
    color = sample_texture(scene_color, scene_sampler, uv)
    # Select rather than branch on a device value, keeping both paths defined.
    result = tonemap(color, post.exposure, False, uv)
    ldr_color = vec4(sm.floor(sm.clamp(color.xyz, 0.0, 1.0) * 255.0 + 0.5) / 255.0, color.w)
    return select((post.comparison > 0.5) & (uv.x < 0.5), ldr_color, result)


def make_post_pipeline(exposure=1.0, comparison=False):
    exposure = validate_exposure(exposure)

    @fragment
    def hdr_post_static(uv, scene_color: Texture2D[0], scene_sampler: Sampler[1]):
        return tonemap(sample_texture(scene_color, scene_sampler, uv), exposure, comparison, uv)

    return Pipeline(fullscreen_uv_vertex, hdr_post_static)
