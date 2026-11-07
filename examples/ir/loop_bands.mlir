module {
  func.func private @tint(%level: f32) -> vector<3xf32> {
    %green = arith.mulf %level, %level : f32
    %half = arith.constant 0.5 : f32
    %quarter = arith.constant 0.25 : f32
    %blue0 = arith.mulf %level, %half : f32
    %blue = arith.addf %quarter, %blue0 : f32
    %color = vector.from_elements %level, %green, %blue : vector<3xf32>
    return %color : vector<3xf32>
  }

  func.func @loop_bands() attributes {gfx.stage = #gfx.stage<fragment>} {
    %pixel = gfx.builtin <fragment_coord> : vector<2xf32>
    %x = vector.extract %pixel[0] : f32 from vector<2xf32>
    %width = arith.constant 256.0 : f32
    %tile = arith.divf %x, %width : f32
    %whole = math.floor %tile : f32
    %fraction = arith.subf %tile, %whole : f32
    %sixteen = arith.constant 16.0 : f32
    %scaled = arith.mulf %fraction, %sixteen : f32
    %count = arith.fptoui %scaled : f32 to i32
    %limit = arith.index_cast %count : i32 to index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %start = arith.constant 0.0 : f32
    %step = arith.constant 0.0625 : f32
    %level = scf.for %i = %zero to %limit step %one iter_args(%value = %start) -> (f32) {
      %next = arith.addf %value, %step : f32
      scf.yield %next : f32
    }
    %color = func.call @tint(%level) : (f32) -> vector<3xf32>
    %r = vector.extract %color[0] : f32 from vector<3xf32>
    %g = vector.extract %color[1] : f32 from vector<3xf32>
    %b = vector.extract %color[2] : f32 from vector<3xf32>
    %alpha = arith.constant 1.0 : f32
    %output = vector.from_elements %r, %g, %b, %alpha : vector<4xf32>
    gfx.output %output 0 "color" <perspective> : vector<4xf32>
    return
  }
}
