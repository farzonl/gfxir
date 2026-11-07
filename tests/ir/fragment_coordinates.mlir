module {
  func.func @coordinates() attributes {gfx.stage = #gfx.stage<fragment>} {
    %xy = gfx.builtin <fragment_coord> : vector<2xf32>
    %x = vector.extract %xy[0] : f32 from vector<2xf32>
    %y = vector.extract %xy[1] : f32 from vector<2xf32>
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %color = vector.from_elements %x, %y, %zero, %one : vector<4xf32>
    gfx.output %color 0 "color" <perspective> : vector<4xf32>
    return
  }
}
