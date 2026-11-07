module {
  func.func @vertex_main() attributes {gfx.stage = #gfx.stage<vertex>} {
    %index = gfx.builtin <vertex_index> : i32
    %zero = arith.constant 0 : i32
    %first = arith.cmpi eq, %index, %zero : i32
    %low = arith.constant -0.5 : f32
    %high = arith.constant 0.5 : f32
    %z = arith.constant 0.0 : f32
    %w = arith.constant 1.0 : f32
    %x = arith.select %first, %low, %high : f32
    %position = vector.from_elements %x, %low, %z, %w : vector<4xf32>
    %uv = vector.from_elements %x, %high : vector<2xf32>
    gfx.position %position : vector<4xf32>
    gfx.output %uv 0 "uv" <perspective> : vector<2xf32>
    return
  }
}
