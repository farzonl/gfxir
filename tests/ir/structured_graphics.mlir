module {
  func.func private @contribution(%i: i32) -> vector<3xf32> {
    %eight = arith.constant 8 : i32
    %first_half = arith.cmpi slt, %i, %eight : i32
    %result = scf.if %first_half -> (vector<3xf32>) {
      %value = arith.constant dense<[0.015625, 0.03125, 0.046875]> : vector<3xf32>
      scf.yield %value : vector<3xf32>
    } else {
      %zero = arith.constant dense<0.0> : vector<3xf32>
      scf.yield %zero : vector<3xf32>
    }
    return %result : vector<3xf32>
  }

  func.func @structured() attributes {gfx.stage = #gfx.stage<fragment>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %sixteen = arith.constant 16 : index
    %zero_color = arith.constant dense<0.0> : vector<3xf32>
    %zero_float = arith.constant 0.0 : f32
    %one_float = arith.constant 1.0 : f32
    %sum:2 = scf.for %i = %zero to %sixteen step %one
        iter_args(%color = %zero_color, %count = %zero_float) -> (vector<3xf32>, f32) {
      %integer = arith.index_cast %i : index to i32
      %value = func.call @contribution(%integer) : (i32) -> vector<3xf32>
      %inner = scf.for %j = %zero to %two step %one iter_args(%current = %color) -> (vector<3xf32>) {
        %next = arith.addf %current, %value : vector<3xf32>
        scf.yield %next : vector<3xf32>
      }
      %next_count = arith.addf %count, %one_float : f32
      scf.yield %inner, %next_count : vector<3xf32>, f32
    }
    %divisor = arith.constant 16.0 : f32
    %alpha = arith.divf %sum#1, %divisor : f32
    %r = vector.extract %sum#0[0] : f32 from vector<3xf32>
    %g = vector.extract %sum#0[1] : f32 from vector<3xf32>
    %b = vector.extract %sum#0[2] : f32 from vector<3xf32>
    %output = vector.from_elements %r, %g, %b, %alpha : vector<4xf32>
    gfx.output %output 0 "color" <perspective> : vector<4xf32>
    return
  }
}
