#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}, gpu.container_module} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}, %arg5: memref<i8>) attributes {gc.num_kernels = 1 : i32} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    gpu.launch_func  @attention_f16_kernel::@attention_f16_kernel blocks in (%c4, %c32, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4x4096x64xf16>, %arg1 : memref<4x4096x64xf16>, %arg2 : memref<4x4096x64xf16>, %arg3 : memref<4x4096x64xf32>)
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
  gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">] {
    gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %0 = ub.poison : f16
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %1 = affine.apply #map()[%block_id_y]
      %subview = memref.subview %arg0[%block_id_x, %1, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %2 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %3:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_3 = memref.subview %arg1[%block_id_x, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %7 = vector.transfer_read %collapse_shape_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %8 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %2, %7, %cst_1 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %9 = vector.multi_reduction <maximumf>, %8, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %10 = arith.maximumf %arg6, %9 : vector<128xf32>
        %11 = vector.shape_cast %10 : vector<128xf32> to vector<128x1xf32>
        %12 = vector.broadcast %11 : vector<128x1xf32> to vector<128x64xf32>
        %13 = arith.subf %8, %12 : vector<128x64xf32>
        %14 = math.exp %13 : vector<128x64xf32>
        %15 = vector.multi_reduction <add>, %14, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %16 = arith.subf %arg6, %10 : vector<128xf32>
        %17 = math.exp %16 : vector<128xf32>
        %18 = arith.mulf %arg7, %17 : vector<128xf32>
        %19 = arith.addf %18, %15 : vector<128xf32>
        %20 = vector.shape_cast %17 : vector<128xf32> to vector<128x1xf32>
        %21 = vector.broadcast %20 : vector<128x1xf32> to vector<128x64xf32>
        %22 = arith.mulf %arg5, %21 : vector<128x64xf32>
        %23 = arith.truncf %14 : vector<128x64xf32> to vector<128x64xf16>
        %subview_5 = memref.subview %arg2[%block_id_x, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %24 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %25 = vector.contract {indexing_maps = [#map1, #map4, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %24, %22 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %25, %10, %19 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %4 = vector.shape_cast %3#2 : vector<128xf32> to vector<128x1xf32>
      %5 = vector.broadcast %4 : vector<128x1xf32> to vector<128x64xf32>
      %6 = arith.divf %3#0, %5 : vector<128x64xf32>
      %subview_2 = memref.subview %arg3[%block_id_x, %1, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      vector.transfer_write %6, %subview_2[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      gpu.return
    }
  }
}