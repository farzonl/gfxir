module {
  func.func @fullscreen() attributes {gfx.stage = #gfx.stage<vertex>} {
    %index = gfx.builtin <vertex_index> : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %is_one = arith.cmpi eq, %index, %one : i32
    %is_two = arith.cmpi eq, %index, %two : i32
    %low = arith.constant -1.0 : f32
    %high = arith.constant 3.0 : f32
    %zero = arith.constant 0.0 : f32
    %w = arith.constant 1.0 : f32
    %x = arith.select %is_one, %high, %low : f32
    %y = arith.select %is_two, %high, %low : f32
    %position = vector.from_elements %x, %y, %zero, %w : vector<4xf32>
    gfx.position %position : vector<4xf32>
    return
  }
}
