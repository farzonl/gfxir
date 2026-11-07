module {
  func.func private @device_types(
      !gfx.ptr<f32, global>,
      !gfx.ptr<vector<4xf32>, workgroup>,
      !gfx.handle<texture_2d>,
      !gfx.handle<sampler>)
}
