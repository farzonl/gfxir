module {
  func.func private @sphere_distance(%point: vector<3xf32>) -> f32 {
    %squared = arith.mulf %point, %point : vector<3xf32>
    %x = vector.extract %squared[0] : f32 from vector<3xf32>
    %y = vector.extract %squared[1] : f32 from vector<3xf32>
    %z = vector.extract %squared[2] : f32 from vector<3xf32>
    %xy = arith.addf %x, %y : f32
    %xyz = arith.addf %xy, %z : f32
    %radius = math.sqrt %xyz : f32
    %size = arith.constant 0.7 : f32
    %distance = arith.subf %radius, %size : f32
    return %distance : f32
  }

  func.func @sphere_march() attributes {gfx.stage = #gfx.stage<fragment>} {
    %pixel = gfx.builtin <fragment_coord> : vector<2xf32>
    %px = vector.extract %pixel[0] : f32 from vector<2xf32>
    %py = vector.extract %pixel[1] : f32 from vector<2xf32>
    %center = arith.constant 128.0 : f32
    %x0 = arith.subf %px, %center : f32
    %y0 = arith.subf %center, %py : f32
    %x = arith.divf %x0, %center : f32
    %y = arith.divf %y0, %center : f32
    %one = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    %direction = vector.from_elements %x, %y, %one : vector<3xf32>
    %length_squared = arith.mulf %direction, %direction : vector<3xf32>
    %lx = vector.extract %length_squared[0] : f32 from vector<3xf32>
    %ly = vector.extract %length_squared[1] : f32 from vector<3xf32>
    %lxy = arith.addf %lx, %ly : f32
    %lxyz = arith.addf %lxy, %one : f32
    %length = math.sqrt %lxyz : f32
    %lengths = vector.broadcast %length : f32 to vector<3xf32>
    %ray = arith.divf %direction, %lengths : vector<3xf32>
    %origin = arith.constant dense<[0.0, 0.0, -3.0]> : vector<3xf32>
    %zero_glow = arith.constant dense<0.0> : vector<3xf32>
    %count0 = arith.constant 0 : i32
    %count1 = arith.constant 1 : i32
    %limit = arith.constant 64 : i32
    %false = arith.constant false
    %max_distance = arith.constant 8.0 : f32
    %threshold = arith.constant 0.001 : f32
    %half = arith.constant 0.5 : f32
    %halves = arith.constant dense<0.5> : vector<3xf32>
    %emission = arith.constant dense<[0.002, 0.004, 0.008]> : vector<3xf32>
    %result:4 = scf.while (%i = %count0, %t = %zero, %hit = %false, %glow = %zero_glow)
        : (i32, f32, i1, vector<3xf32>) -> (i32, f32, i1, vector<3xf32>) {
      %within_steps = arith.cmpi slt, %i, %limit : i32
      %within_distance = arith.cmpf ole, %t, %max_distance : f32
      %not_hit = arith.cmpi eq, %hit, %false : i1
      %running0 = arith.andi %within_steps, %within_distance : i1
      %running = arith.andi %running0, %not_hit : i1
      scf.condition(%running) %i, %t, %hit, %glow : i32, f32, i1, vector<3xf32>
    } do {
    ^bb0(%i: i32, %t: f32, %hit: i1, %glow: vector<3xf32>):
      %ts = vector.broadcast %t : f32 to vector<3xf32>
      %offset = arith.mulf %ray, %ts : vector<3xf32>
      %point = arith.addf %origin, %offset : vector<3xf32>
      %distance = func.call @sphere_distance(%point) : (vector<3xf32>) -> f32
      %new_hit = arith.cmpf olt, %distance, %threshold : f32
      %next_t = scf.if %new_hit -> (f32) {
        scf.yield %t : f32
      } else {
        %advance = arith.mulf %distance, %half : f32
        %next = arith.addf %t, %advance : f32
        scf.yield %next : f32
      }
      %next_i = arith.addi %i, %count1 : i32
      %next_glow = arith.addf %glow, %emission : vector<3xf32>
      scf.yield %next_i, %next_t, %new_hit, %next_glow : i32, f32, i1, vector<3xf32>
    }
    %ts = vector.broadcast %result#1 : f32 to vector<3xf32>
    %offset = arith.mulf %ray, %ts : vector<3xf32>
    %point = arith.addf %origin, %offset : vector<3xf32>
    %size = arith.constant dense<0.7> : vector<3xf32>
    %normal = arith.divf %point, %size : vector<3xf32>
    %normal_color0 = arith.mulf %normal, %halves : vector<3xf32>
    %normal_color = arith.addf %normal_color0, %halves : vector<3xf32>
    %background = arith.constant dense<[0.02, 0.04, 0.08]> : vector<3xf32>
    %color = arith.select %result#2, %normal_color, %background : vector<3xf32>
    %lit = arith.addf %color, %result#3 : vector<3xf32>
    %r = vector.extract %lit[0] : f32 from vector<3xf32>
    %g = vector.extract %lit[1] : f32 from vector<3xf32>
    %b = vector.extract %lit[2] : f32 from vector<3xf32>
    %output = vector.from_elements %r, %g, %b, %one : vector<4xf32>
    gfx.output %output 0 "color" <perspective> : vector<4xf32>
    return
  }
}
