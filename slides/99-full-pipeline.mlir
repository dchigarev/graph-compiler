// -----// IR Dump After GpuDeviceProps (gpu-dev-props) //----- //
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = linalgx.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%arg0, %arg1, %arg2, %cst : tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, f16) outs(%arg3 : tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32>
    return %0 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump After GpuDeviceProps (gpu-dev-props) //----- //
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = linalgx.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%arg0, %arg1, %arg2, %cst : tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, f16) outs(%arg3 : tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32>
    return %0 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = linalgx.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%arg0, %arg1, %arg2, %cst : tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, f16) outs(%arg3 : tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32>
    return %0 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = linalgx.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%arg0, %arg1, %arg2, %cst : tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, f16) outs(%arg3 : tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32>
    return %0 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump Initial //----- //
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = linalgx.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%arg0, %arg1, %arg2, %cst : tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, f16) outs(%arg3 : tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32>
    return %0 : tensor<4x4096x64xf32>
  }
}
// -----// IR Dump After PrintIRPass (print-ir) //----- //
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = linalgx.attention {indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%arg0, %arg1, %arg2, %cst : tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, f16) outs(%arg3 : tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32>
    return %0 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump After LinalgElementwiseOpFusionPass (linalg-fuse-elementwise-ops) //----- //
func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = linalgx.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%arg0, %arg1, %arg2, %cst : tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, f16) outs(%arg3 : tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32>
  return %0 : tensor<4x4096x64xf32>
}

// -----// IR Dump After TileContraction (tile-contract) //----- //
func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = linalgx.attention {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%arg0, %arg1, %arg2, %cst : tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, f16) outs(%arg3 : tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32>
  return %0 : tensor<4x4096x64xf32>
}

AttentionOp iteration domain: 
  Dim 0: offset=0 : index, size=4 : index, stride=1 : index
  Dim 1: offset=0 : index, size=4096 : index, stride=1 : index
  Dim 2: offset=0 : index, size=64 : index, stride=1 : index
  Dim 3: offset=0 : index, size=4096 : index, stride=1 : index
  Dim 4: offset=0 : index, size=64 : index, stride=1 : index
indexing maps: (d0, d1, d2, d3, d4) -> (d0, d1, d2), (d0, d1, d2, d3, d4) -> (d0, d3, d2), (d0, d1, d2, d3, d4) -> (d0, d3, d4), (d0, d1, d2, d3, d4) -> (d0, d1, d4)
AttentionOp iterator types: 
  Dim 0: parallel
  Dim 1: parallel
  Dim 2: reduction
  Dim 3: reduction
  Dim 4: parallel
AttentionOp iteration domain: 
  Dim 0: offset=0 : index, size=4 : index, stride=1 : index
  Dim 1: offset=0 : index, size=4096 : index, stride=1 : index
  Dim 2: offset=0 : index, size=64 : index, stride=1 : index
  Dim 3: offset=0 : index, size=4096 : index, stride=1 : index
  Dim 4: offset=0 : index, size=64 : index, stride=1 : index
indexing maps: (d0, d1, d2, d3, d4) -> (d0, d1, d2), (d0, d1, d2, d3, d4) -> (d0, d3, d2), (d0, d1, d2, d3, d4) -> (d0, d3, d4), (d0, d1, d2, d3, d4) -> (d0, d1, d4)
AttentionOp iterator types: 
  Dim 0: parallel
  Dim 1: parallel
  Dim 2: reduction
  Dim 3: reduction
  Dim 4: parallel
AttentionOp iterator types: 
  Dim 0: parallel
  Dim 1: parallel
  Dim 2: reduction
  Dim 3: reduction
  Dim 4: parallel
AttentionOp iteration domain: 
  Dim 0: offset=0 : index, size=1 : index, stride=1 : index
  Dim 1: offset=0 : index, size=128 : index, stride=1 : index
  Dim 2: offset=0 : index, size=64 : index, stride=1 : index
  Dim 3: offset=0 : index, size=4096 : index, stride=1 : index
  Dim 4: offset=0 : index, size=64 : index, stride=1 : index
indexing maps: (d0, d1, d2, d3, d4) -> (d0, d1, d2), (d0, d1, d2, d3, d4) -> (d0, d3, d2), (d0, d1, d2, d3, d4) -> (d0, d3, d4), (d0, d1, d2, d3, d4) -> (d0, d1, d4)
AttentionOp iterator types: 
  Dim 0: parallel
  Dim 1: parallel
  Dim 2: reduction
  Dim 3: reduction
  Dim 4: parallel
// -----// IR Dump After TileAttention (tile-attention) //----- //
func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %extracted_slice_1 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %extracted_slice_2 = tensor.extract_slice %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf32> to tensor<1x128x64xf32>
    %1 = linalgx.attention {gc.tiling.level = 1 : i8, gc.tiling.mode = 0 : i8, indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %cst : tensor<1x128x64xf16>, tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, f16) outs(%extracted_slice_2 : tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %1 into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
    }
  } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
  return %0 : tensor<4x4096x64xf32>
}

// -----// IR Dump After TileParallel (tile-parallel) //----- //
func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %extracted_slice_1 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %extracted_slice_2 = tensor.extract_slice %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf32> to tensor<1x128x64xf32>
    %1 = linalgx.attention {gc.tiling.level = 1 : i8, gc.tiling.mode = 0 : i8, indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %cst : tensor<1x128x64xf16>, tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, f16) outs(%extracted_slice_2 : tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %1 into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
    }
  } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
  return %0 : tensor<4x4096x64xf32>
}

// -----// IR Dump After CSEPass (cse) //----- //
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_2 = tensor.extract_slice %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf32> to tensor<1x128x64xf32>
      %1 = linalgx.attention {gc.tiling.level = 1 : i8, gc.tiling.mode = 0 : i8, indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %cst : tensor<1x128x64xf16>, tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, f16) outs(%extracted_slice_2 : tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %1 into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
      }
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %0 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_2 = tensor.extract_slice %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf32> to tensor<1x128x64xf32>
      %1 = linalgx.attention {gc.tiling.level = 1 : i8, gc.tiling.mode = 0 : i8, indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %cst : tensor<1x128x64xf16>, tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, f16) outs(%extracted_slice_2 : tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %1 into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
      }
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %0 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump Tiling //----- //
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_2 = tensor.extract_slice %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf32> to tensor<1x128x64xf32>
      %1 = linalgx.attention {gc.tiling.level = 1 : i8, gc.tiling.mode = 0 : i8, indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> ()>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>]} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %cst : tensor<1x128x64xf16>, tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, f16) outs(%extracted_slice_2 : tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %1 into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
      }
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %0 : tensor<4x4096x64xf32>
  }
}
// -----// IR Dump After PrintIRPass (print-ir) //----- //
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_2 = tensor.extract_slice %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf32> to tensor<1x128x64xf32>
      %1 = linalgx.attention {gc.tiling.level = 1 : i8, gc.tiling.mode = 0 : i8, indexing_maps = [#map, #map1, #map2, #map3, #map4]} ins(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %cst : tensor<1x128x64xf16>, tensor<1x4096x64xf16>, tensor<1x4096x64xf16>, f16) outs(%extracted_slice_2 : tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %1 into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
      }
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %0 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump After Decomposition (decomposition) //----- //
func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
    %extracted_slice_1 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %extracted_slice_2 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x128x64xf16> into tensor<128x64xf16>
    %1 = tensor.empty() : tensor<128xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<128xf32>) -> tensor<128xf32>
    %3 = tensor.empty() : tensor<128xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128xf32>) -> tensor<128xf32>
    %5 = tensor.empty() : tensor<128x64xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %7:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %6, %arg9 = %2, %arg10 = %4) -> (tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>) {
      %extracted_slice_3 = tensor.extract_slice %extracted_slice_1[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_4 = tensor.collapse_shape %extracted_slice_3 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %11 = tensor.empty() : tensor<64x64xf16>
      %transposed = linalg.transpose ins(%collapsed_4 : tensor<64x64xf16>) outs(%11 : tensor<64x64xf16>) permutation = [1, 0] 
      %12 = tensor.empty() : tensor<128x64xf32>
      %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %14 = linalg.matmul ins(%collapsed, %transposed : tensor<128x64xf16>, tensor<64x64xf16>) outs(%13 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %15 = tensor.empty() : tensor<128xf32>
      %16 = linalg.fill ins(%cst_0 : f32) outs(%15 : tensor<128xf32>) -> tensor<128xf32>
      %reduced = linalg.reduce ins(%14 : tensor<128x64xf32>) outs(%16 : tensor<128xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %40 = arith.maximumf %in, %init : f32
          linalg.yield %40 : f32
        }
      %17 = tensor.empty() : tensor<128xf32>
      %18 = linalg.max ins(%arg9, %reduced : tensor<128xf32>, tensor<128xf32>) outs(%17 : tensor<128xf32>) -> tensor<128xf32>
      %19 = tensor.empty() : tensor<128x64xf32>
      %broadcasted_5 = linalg.broadcast ins(%18 : tensor<128xf32>) outs(%19 : tensor<128x64xf32>) dimensions = [1] 
      %20 = tensor.empty() : tensor<128x64xf32>
      %21 = linalg.sub ins(%14, %broadcasted_5 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%20 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %22 = tensor.empty() : tensor<128x64xf32>
      %23 = linalg.exp ins(%21 : tensor<128x64xf32>) outs(%22 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %24 = tensor.empty() : tensor<128xf32>
      %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<128xf32>) -> tensor<128xf32>
      %reduced_6 = linalg.reduce ins(%23 : tensor<128x64xf32>) outs(%25 : tensor<128xf32>) dimensions = [1] 
        (%in: f32, %init: f32) {
          %40 = arith.addf %in, %init : f32
          linalg.yield %40 : f32
        }
      %26 = tensor.empty() : tensor<128xf32>
      %27 = linalg.sub ins(%arg9, %18 : tensor<128xf32>, tensor<128xf32>) outs(%26 : tensor<128xf32>) -> tensor<128xf32>
      %28 = tensor.empty() : tensor<128xf32>
      %29 = linalg.exp ins(%27 : tensor<128xf32>) outs(%28 : tensor<128xf32>) -> tensor<128xf32>
      %30 = tensor.empty() : tensor<128xf32>
      %31 = linalg.mul ins(%arg10, %29 : tensor<128xf32>, tensor<128xf32>) outs(%30 : tensor<128xf32>) -> tensor<128xf32>
      %32 = tensor.empty() : tensor<128xf32>
      %33 = linalg.add ins(%31, %reduced_6 : tensor<128xf32>, tensor<128xf32>) outs(%32 : tensor<128xf32>) -> tensor<128xf32>
      %34 = tensor.empty() : tensor<128x64xf32>
      %broadcasted_7 = linalg.broadcast ins(%29 : tensor<128xf32>) outs(%34 : tensor<128x64xf32>) dimensions = [1] 
      %35 = tensor.empty() : tensor<128x64xf32>
      %36 = linalg.mul ins(%arg8, %broadcasted_7 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%35 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %37 = tensor.empty() : tensor<128x64xf16>
      %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23 : tensor<128x64xf32>) outs(%37 : tensor<128x64xf16>) {
      ^bb0(%in: f32, %out: f16):
        %40 = arith.truncf %in : f32 to f16
        linalg.yield %40 : f16
      } -> tensor<128x64xf16>
      %extracted_slice_8 = tensor.extract_slice %extracted_slice_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_9 = tensor.collapse_shape %extracted_slice_8 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %39 = linalg.matmul ins(%38, %collapsed_9 : tensor<128x64xf16>, tensor<64x64xf16>) outs(%36 : tensor<128x64xf32>) -> tensor<128x64xf32>
      scf.yield %39, %18, %33 : tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>
    }
    %8 = tensor.empty() : tensor<128x64xf32>
    %broadcasted = linalg.broadcast ins(%7#2 : tensor<128xf32>) outs(%8 : tensor<128x64xf32>) dimensions = [1] 
    %9 = tensor.empty() : tensor<128x64xf32>
    %10 = linalg.div ins(%7#0, %broadcasted : tensor<128x64xf32>, tensor<128x64xf32>) outs(%9 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %expanded = tensor.expand_shape %10 [[0, 1], [2]] output_shape [1, 128, 64] : tensor<128x64xf32> into tensor<1x128x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %expanded into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
    }
  } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
  return %0 : tensor<4x4096x64xf32>
}

// -----// IR Dump After Vectorize (vectorize) //----- //
func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
  %0 = ub.poison : f32
  %1 = ub.poison : f16
  %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
  %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %2 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
    %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %extracted_slice_3 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x128x64xf16> into tensor<128x64xf16>
    %3 = tensor.empty() : tensor<128xf32>
    %4 = vector.transfer_write %cst_1, %3[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
    %5 = tensor.empty() : tensor<128xf32>
    %6 = vector.transfer_write %cst_0, %5[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
    %7 = tensor.empty() : tensor<128x64xf32>
    %8 = vector.transfer_write %cst, %7[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
    %9:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %8, %arg9 = %4, %arg10 = %6) -> (tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>) {
      %extracted_slice_4 = tensor.extract_slice %extracted_slice_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_5 = tensor.collapse_shape %extracted_slice_4 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %17 = vector.transfer_read %collapsed_5[%c0, %c0], %1 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
      %18 = vector.transfer_read %collapsed[%c0, %c0], %1 {in_bounds = [true, true]} : tensor<128x64xf16>, vector<128x64xf16>
      %19 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %17, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %20 = vector.multi_reduction <maximumf>, %19, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
      %21 = tensor.empty() : tensor<128xf32>
      %22 = vector.transfer_read %arg9[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
      %23 = arith.maximumf %22, %20 : vector<128xf32>
      %24 = vector.transfer_write %23, %21[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
      %25 = vector.shape_cast %23 : vector<128xf32> to vector<128x1xf32>
      %26 = vector.broadcast %25 : vector<128x1xf32> to vector<128x64xf32>
      %27 = arith.subf %19, %26 : vector<128x64xf32>
      %28 = math.exp %27 : vector<128x64xf32>
      %29 = vector.multi_reduction <add>, %28, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
      %30 = vector.transfer_read %arg9[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
      %31 = arith.subf %30, %23 : vector<128xf32>
      %32 = math.exp %31 : vector<128xf32>
      %33 = vector.transfer_read %arg10[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
      %34 = arith.mulf %33, %32 : vector<128xf32>
      %35 = tensor.empty() : tensor<128xf32>
      %36 = arith.addf %34, %29 : vector<128xf32>
      %37 = vector.transfer_write %36, %35[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
      %38 = vector.shape_cast %32 : vector<128xf32> to vector<128x1xf32>
      %39 = vector.broadcast %38 : vector<128x1xf32> to vector<128x64xf32>
      %40 = tensor.empty() : tensor<128x64xf32>
      %41 = vector.transfer_read %arg8[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf32>, vector<128x64xf32>
      %42 = arith.mulf %41, %39 : vector<128x64xf32>
      %43 = arith.truncf %28 : vector<128x64xf32> to vector<128x64xf16>
      %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %44 = vector.transfer_read %collapsed_7[%c0, %c0], %1 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
      %45 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %44, %42 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %46 = vector.transfer_write %45, %40[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
      scf.yield %46, %24, %37 : tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>
    }
    %10 = vector.transfer_read %9#2[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
    %11 = vector.shape_cast %10 : vector<128xf32> to vector<128x1xf32>
    %12 = vector.broadcast %11 : vector<128x1xf32> to vector<128x64xf32>
    %13 = tensor.empty() : tensor<128x64xf32>
    %14 = vector.transfer_read %9#0[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf32>, vector<128x64xf32>
    %15 = arith.divf %14, %12 : vector<128x64xf32>
    %16 = vector.transfer_write %15, %13[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
    %expanded = tensor.expand_shape %16 [[0, 1], [2]] output_shape [1, 128, 64] : tensor<128x64xf32> into tensor<1x128x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %expanded into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
    }
  } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
  return %2 : tensor<4x4096x64xf32>
}

// -----// IR Dump After LoopInvariantCodeMotionPass (loop-invariant-code-motion) //----- //
func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
  %0 = ub.poison : f32
  %1 = ub.poison : f16
  %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
  %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %2 = tensor.empty() : tensor<128xf32>
  %3 = vector.transfer_write %cst_1, %2[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
  %4 = tensor.empty() : tensor<128xf32>
  %5 = vector.transfer_write %cst_0, %4[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
  %6 = tensor.empty() : tensor<128x64xf32>
  %7 = vector.transfer_write %cst, %6[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
  %8 = tensor.empty() : tensor<128xf32>
  %9 = tensor.empty() : tensor<128xf32>
  %10 = tensor.empty() : tensor<128x64xf32>
  %11 = tensor.empty() : tensor<128x64xf32>
  %12 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
    %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %extracted_slice_3 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x128x64xf16> into tensor<128x64xf16>
    %13 = vector.transfer_read %collapsed[%c0, %c0], %1 {in_bounds = [true, true]} : tensor<128x64xf16>, vector<128x64xf16>
    %14:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %7, %arg9 = %3, %arg10 = %5) -> (tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>) {
      %extracted_slice_4 = tensor.extract_slice %extracted_slice_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_5 = tensor.collapse_shape %extracted_slice_4 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %21 = vector.transfer_read %collapsed_5[%c0, %c0], %1 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
      %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %13, %21, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %23 = vector.multi_reduction <maximumf>, %22, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
      %24 = vector.transfer_read %arg9[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
      %25 = arith.maximumf %24, %23 : vector<128xf32>
      %26 = vector.transfer_write %25, %8[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
      %27 = vector.shape_cast %25 : vector<128xf32> to vector<128x1xf32>
      %28 = vector.broadcast %27 : vector<128x1xf32> to vector<128x64xf32>
      %29 = arith.subf %22, %28 : vector<128x64xf32>
      %30 = math.exp %29 : vector<128x64xf32>
      %31 = vector.multi_reduction <add>, %30, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
      %32 = vector.transfer_read %arg9[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
      %33 = arith.subf %32, %25 : vector<128xf32>
      %34 = math.exp %33 : vector<128xf32>
      %35 = vector.transfer_read %arg10[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
      %36 = arith.mulf %35, %34 : vector<128xf32>
      %37 = arith.addf %36, %31 : vector<128xf32>
      %38 = vector.transfer_write %37, %9[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
      %39 = vector.shape_cast %34 : vector<128xf32> to vector<128x1xf32>
      %40 = vector.broadcast %39 : vector<128x1xf32> to vector<128x64xf32>
      %41 = vector.transfer_read %arg8[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf32>, vector<128x64xf32>
      %42 = arith.mulf %41, %40 : vector<128x64xf32>
      %43 = arith.truncf %30 : vector<128x64xf32> to vector<128x64xf16>
      %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %44 = vector.transfer_read %collapsed_7[%c0, %c0], %1 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
      %45 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %44, %42 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %46 = vector.transfer_write %45, %10[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
      scf.yield %46, %26, %38 : tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>
    }
    %15 = vector.transfer_read %14#2[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
    %16 = vector.shape_cast %15 : vector<128xf32> to vector<128x1xf32>
    %17 = vector.broadcast %16 : vector<128x1xf32> to vector<128x64xf32>
    %18 = vector.transfer_read %14#0[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf32>, vector<128x64xf32>
    %19 = arith.divf %18, %17 : vector<128x64xf32>
    %20 = vector.transfer_write %19, %11[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
    %expanded = tensor.expand_shape %20 [[0, 1], [2]] output_shape [1, 128, 64] : tensor<128x64xf32> into tensor<1x128x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %expanded into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
    }
  } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
  return %12 : tensor<4x4096x64xf32>
}

// -----// IR Dump After LoopInvariantSubsetHoistingPass (loop-invariant-subset-hoisting) //----- //
func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
  %0 = ub.poison : f32
  %1 = ub.poison : f16
  %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
  %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %2 = tensor.empty() : tensor<128xf32>
  %3 = vector.transfer_write %cst_1, %2[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
  %4 = tensor.empty() : tensor<128xf32>
  %5 = vector.transfer_write %cst_0, %4[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
  %6 = tensor.empty() : tensor<128x64xf32>
  %7 = vector.transfer_write %cst, %6[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
  %8 = tensor.empty() : tensor<128xf32>
  %9 = tensor.empty() : tensor<128xf32>
  %10 = tensor.empty() : tensor<128x64xf32>
  %11 = tensor.empty() : tensor<128x64xf32>
  %12 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
    %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %extracted_slice_3 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x128x64xf16> into tensor<128x64xf16>
    %13 = vector.transfer_read %collapsed[%c0, %c0], %1 {in_bounds = [true, true]} : tensor<128x64xf16>, vector<128x64xf16>
    %14:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %7, %arg9 = %3, %arg10 = %5) -> (tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>) {
      %extracted_slice_4 = tensor.extract_slice %extracted_slice_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_5 = tensor.collapse_shape %extracted_slice_4 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %21 = vector.transfer_read %collapsed_5[%c0, %c0], %1 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
      %22 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %13, %21, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %23 = vector.multi_reduction <maximumf>, %22, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
      %24 = vector.transfer_read %arg9[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
      %25 = arith.maximumf %24, %23 : vector<128xf32>
      %26 = vector.transfer_write %25, %8[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
      %27 = vector.shape_cast %25 : vector<128xf32> to vector<128x1xf32>
      %28 = vector.broadcast %27 : vector<128x1xf32> to vector<128x64xf32>
      %29 = arith.subf %22, %28 : vector<128x64xf32>
      %30 = math.exp %29 : vector<128x64xf32>
      %31 = vector.multi_reduction <add>, %30, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
      %32 = vector.transfer_read %arg9[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
      %33 = arith.subf %32, %25 : vector<128xf32>
      %34 = math.exp %33 : vector<128xf32>
      %35 = vector.transfer_read %arg10[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
      %36 = arith.mulf %35, %34 : vector<128xf32>
      %37 = arith.addf %36, %31 : vector<128xf32>
      %38 = vector.transfer_write %37, %9[%c0] {in_bounds = [true]} : vector<128xf32>, tensor<128xf32>
      %39 = vector.shape_cast %34 : vector<128xf32> to vector<128x1xf32>
      %40 = vector.broadcast %39 : vector<128x1xf32> to vector<128x64xf32>
      %41 = vector.transfer_read %arg8[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf32>, vector<128x64xf32>
      %42 = arith.mulf %41, %40 : vector<128x64xf32>
      %43 = arith.truncf %30 : vector<128x64xf32> to vector<128x64xf16>
      %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %44 = vector.transfer_read %collapsed_7[%c0, %c0], %1 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
      %45 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %44, %42 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %46 = vector.transfer_write %45, %10[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
      scf.yield %46, %26, %38 : tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>
    }
    %15 = vector.transfer_read %14#2[%c0], %0 {in_bounds = [true]} : tensor<128xf32>, vector<128xf32>
    %16 = vector.shape_cast %15 : vector<128xf32> to vector<128x1xf32>
    %17 = vector.broadcast %16 : vector<128x1xf32> to vector<128x64xf32>
    %18 = vector.transfer_read %14#0[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf32>, vector<128x64xf32>
    %19 = arith.divf %18, %17 : vector<128x64xf32>
    %20 = vector.transfer_write %19, %11[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
    %expanded = tensor.expand_shape %20 [[0, 1], [2]] output_shape [1, 128, 64] : tensor<128x64xf32> into tensor<1x128x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %expanded into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
    }
  } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
  return %12 : tensor<4x4096x64xf32>
}

// -----// IR Dump After HoistForLoopTransferRead (hoist-for-loop-transfer-read) //----- //
func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
  %0 = ub.poison : f16
  %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
  %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %1 = tensor.empty() : tensor<128x64xf32>
  %2 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
    %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %extracted_slice_3 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x128x64xf16> into tensor<128x64xf16>
    %3 = vector.transfer_read %collapsed[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf16>, vector<128x64xf16>
    %4:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      %extracted_slice_4 = tensor.extract_slice %extracted_slice_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_5 = tensor.collapse_shape %extracted_slice_4 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %9 = vector.transfer_read %collapsed_5[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
      %10 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %9, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %11 = vector.multi_reduction <maximumf>, %10, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
      %12 = arith.maximumf %arg9, %11 : vector<128xf32>
      %13 = vector.shape_cast %12 : vector<128xf32> to vector<128x1xf32>
      %14 = vector.broadcast %13 : vector<128x1xf32> to vector<128x64xf32>
      %15 = arith.subf %10, %14 : vector<128x64xf32>
      %16 = math.exp %15 : vector<128x64xf32>
      %17 = vector.multi_reduction <add>, %16, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
      %18 = arith.subf %arg9, %12 : vector<128xf32>
      %19 = math.exp %18 : vector<128xf32>
      %20 = arith.mulf %arg10, %19 : vector<128xf32>
      %21 = arith.addf %20, %17 : vector<128xf32>
      %22 = vector.shape_cast %19 : vector<128xf32> to vector<128x1xf32>
      %23 = vector.broadcast %22 : vector<128x1xf32> to vector<128x64xf32>
      %24 = arith.mulf %arg8, %23 : vector<128x64xf32>
      %25 = arith.truncf %16 : vector<128x64xf32> to vector<128x64xf16>
      %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
      %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
      %26 = vector.transfer_read %collapsed_7[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
      %27 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %25, %26, %24 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      scf.yield %27, %12, %21 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    }
    %5 = vector.shape_cast %4#2 : vector<128xf32> to vector<128x1xf32>
    %6 = vector.broadcast %5 : vector<128x1xf32> to vector<128x64xf32>
    %7 = arith.divf %4#0, %6 : vector<128x64xf32>
    %8 = vector.transfer_write %7, %1[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
    %expanded = tensor.expand_shape %8 [[0, 1], [2]] output_shape [1, 128, 64] : tensor<128x64xf32> into tensor<1x128x64xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %expanded into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
    }
  } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
  return %2 : tensor<4x4096x64xf32>
}

// -----// IR Dump After CSEPass (cse) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<128x64xf32>
    %2 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
      %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_3 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x128x64xf16> into tensor<128x64xf16>
      %3 = vector.transfer_read %collapsed[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf16>, vector<128x64xf16>
      %4:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
        %collapsed_5 = tensor.collapse_shape %extracted_slice_4 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
        %9 = vector.transfer_read %collapsed_5[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
        %10 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %9, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %11 = vector.multi_reduction <maximumf>, %10, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %12 = arith.maximumf %arg9, %11 : vector<128xf32>
        %13 = vector.shape_cast %12 : vector<128xf32> to vector<128x1xf32>
        %14 = vector.broadcast %13 : vector<128x1xf32> to vector<128x64xf32>
        %15 = arith.subf %10, %14 : vector<128x64xf32>
        %16 = math.exp %15 : vector<128x64xf32>
        %17 = vector.multi_reduction <add>, %16, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %18 = arith.subf %arg9, %12 : vector<128xf32>
        %19 = math.exp %18 : vector<128xf32>
        %20 = arith.mulf %arg10, %19 : vector<128xf32>
        %21 = arith.addf %20, %17 : vector<128xf32>
        %22 = vector.shape_cast %19 : vector<128xf32> to vector<128x1xf32>
        %23 = vector.broadcast %22 : vector<128x1xf32> to vector<128x64xf32>
        %24 = arith.mulf %arg8, %23 : vector<128x64xf32>
        %25 = arith.truncf %16 : vector<128x64xf32> to vector<128x64xf16>
        %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
        %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
        %26 = vector.transfer_read %collapsed_7[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
        %27 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %25, %26, %24 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %27, %12, %21 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %5 = vector.shape_cast %4#2 : vector<128xf32> to vector<128x1xf32>
      %6 = vector.broadcast %5 : vector<128x1xf32> to vector<128x64xf32>
      %7 = arith.divf %4#0, %6 : vector<128x64xf32>
      %8 = vector.transfer_write %7, %1[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
      %expanded = tensor.expand_shape %8 [[0, 1], [2]] output_shape [1, 128, 64] : tensor<128x64xf32> into tensor<1x128x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %expanded into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
      }
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %2 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<128x64xf32>
    %2 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
      %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_3 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x128x64xf16> into tensor<128x64xf16>
      %3 = vector.transfer_read %collapsed[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf16>, vector<128x64xf16>
      %4:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
        %collapsed_5 = tensor.collapse_shape %extracted_slice_4 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
        %9 = vector.transfer_read %collapsed_5[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
        %10 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %9, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %11 = vector.multi_reduction <maximumf>, %10, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %12 = arith.maximumf %arg9, %11 : vector<128xf32>
        %13 = vector.shape_cast %12 : vector<128xf32> to vector<128x1xf32>
        %14 = vector.broadcast %13 : vector<128x1xf32> to vector<128x64xf32>
        %15 = arith.subf %10, %14 : vector<128x64xf32>
        %16 = math.exp %15 : vector<128x64xf32>
        %17 = vector.multi_reduction <add>, %16, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %18 = arith.subf %arg9, %12 : vector<128xf32>
        %19 = math.exp %18 : vector<128xf32>
        %20 = arith.mulf %arg10, %19 : vector<128xf32>
        %21 = arith.addf %20, %17 : vector<128xf32>
        %22 = vector.shape_cast %19 : vector<128xf32> to vector<128x1xf32>
        %23 = vector.broadcast %22 : vector<128x1xf32> to vector<128x64xf32>
        %24 = arith.mulf %arg8, %23 : vector<128x64xf32>
        %25 = arith.truncf %16 : vector<128x64xf32> to vector<128x64xf16>
        %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
        %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
        %26 = vector.transfer_read %collapsed_7[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
        %27 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %25, %26, %24 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %27, %12, %21 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %5 = vector.shape_cast %4#2 : vector<128xf32> to vector<128x1xf32>
      %6 = vector.broadcast %5 : vector<128x1xf32> to vector<128x64xf32>
      %7 = arith.divf %4#0, %6 : vector<128x64xf32>
      %8 = vector.transfer_write %7, %1[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
      %expanded = tensor.expand_shape %8 [[0, 1], [2]] output_shape [1, 128, 64] : tensor<128x64xf32> into tensor<1x128x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %expanded into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
      }
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %2 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump Vectorization //----- //
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<128x64xf32>
    %2 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
      %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_3 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x128x64xf16> into tensor<128x64xf16>
      %3 = vector.transfer_read %collapsed[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf16>, vector<128x64xf16>
      %4:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
        %collapsed_5 = tensor.collapse_shape %extracted_slice_4 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
        %9 = vector.transfer_read %collapsed_5[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
        %10 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %9, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %11 = vector.multi_reduction <maximumf>, %10, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %12 = arith.maximumf %arg9, %11 : vector<128xf32>
        %13 = vector.shape_cast %12 : vector<128xf32> to vector<128x1xf32>
        %14 = vector.broadcast %13 : vector<128x1xf32> to vector<128x64xf32>
        %15 = arith.subf %10, %14 : vector<128x64xf32>
        %16 = math.exp %15 : vector<128x64xf32>
        %17 = vector.multi_reduction <add>, %16, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %18 = arith.subf %arg9, %12 : vector<128xf32>
        %19 = math.exp %18 : vector<128xf32>
        %20 = arith.mulf %arg10, %19 : vector<128xf32>
        %21 = arith.addf %20, %17 : vector<128xf32>
        %22 = vector.shape_cast %19 : vector<128xf32> to vector<128x1xf32>
        %23 = vector.broadcast %22 : vector<128x1xf32> to vector<128x64xf32>
        %24 = arith.mulf %arg8, %23 : vector<128x64xf32>
        %25 = arith.truncf %16 : vector<128x64xf32> to vector<128x64xf16>
        %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
        %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
        %26 = vector.transfer_read %collapsed_7[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
        %27 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %25, %26, %24 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %27, %12, %21 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %5 = vector.shape_cast %4#2 : vector<128xf32> to vector<128x1xf32>
      %6 = vector.broadcast %5 : vector<128x1xf32> to vector<128x64xf32>
      %7 = arith.divf %4#0, %6 : vector<128x64xf32>
      %8 = vector.transfer_write %7, %1[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
      %expanded = tensor.expand_shape %8 [[0, 1], [2]] output_shape [1, 128, 64] : tensor<128x64xf32> into tensor<1x128x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %expanded into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
      }
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %2 : tensor<4x4096x64xf32>
  }
}
// -----// IR Dump After PrintIRPass (print-ir) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>, %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %1 = tensor.empty() : tensor<128x64xf32>
    %2 = scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) shared_outs(%arg6 = %arg3) -> (tensor<4x4096x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x128x64xf16>
      %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %extracted_slice_3 = tensor.extract_slice %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : tensor<4x4096x64xf16> to tensor<1x4096x64xf16>
      %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x128x64xf16> into tensor<128x64xf16>
      %3 = vector.transfer_read %collapsed[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<128x64xf16>, vector<128x64xf16>
      %4:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
        %collapsed_5 = tensor.collapse_shape %extracted_slice_4 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
        %9 = vector.transfer_read %collapsed_5[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
        %10 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %9, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %11 = vector.multi_reduction <maximumf>, %10, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %12 = arith.maximumf %arg9, %11 : vector<128xf32>
        %13 = vector.shape_cast %12 : vector<128xf32> to vector<128x1xf32>
        %14 = vector.broadcast %13 : vector<128x1xf32> to vector<128x64xf32>
        %15 = arith.subf %10, %14 : vector<128x64xf32>
        %16 = math.exp %15 : vector<128x64xf32>
        %17 = vector.multi_reduction <add>, %16, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %18 = arith.subf %arg9, %12 : vector<128xf32>
        %19 = math.exp %18 : vector<128xf32>
        %20 = arith.mulf %arg10, %19 : vector<128xf32>
        %21 = arith.addf %20, %17 : vector<128xf32>
        %22 = vector.shape_cast %19 : vector<128xf32> to vector<128x1xf32>
        %23 = vector.broadcast %22 : vector<128x1xf32> to vector<128x64xf32>
        %24 = arith.mulf %arg8, %23 : vector<128x64xf32>
        %25 = arith.truncf %16 : vector<128x64xf32> to vector<128x64xf16>
        %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : tensor<1x4096x64xf16> to tensor<1x64x64xf16>
        %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [[0, 1], [2]] : tensor<1x64x64xf16> into tensor<64x64xf16>
        %26 = vector.transfer_read %collapsed_7[%c0, %c0], %0 {in_bounds = [true, true]} : tensor<64x64xf16>, vector<64x64xf16>
        %27 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %25, %26, %24 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %27, %12, %21 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %5 = vector.shape_cast %4#2 : vector<128xf32> to vector<128x1xf32>
      %6 = vector.broadcast %5 : vector<128x1xf32> to vector<128x64xf32>
      %7 = arith.divf %4#0, %6 : vector<128x64xf32>
      %8 = vector.transfer_write %7, %1[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, tensor<128x64xf32>
      %expanded = tensor.expand_shape %8 [[0, 1], [2]] output_shape [1, 128, 64] : tensor<128x64xf32> into tensor<1x128x64xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %expanded into %arg6[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : tensor<1x128x64xf32> into tensor<4x4096x64xf32>
      }
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %2 : tensor<4x4096x64xf32>
  }
}


// -----// IR Dump After OneShotBufferizePass (one-shot-bufferize) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) -> memref<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_3 = memref.subview %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg6 = %c0 to %c4096 step %c64 iter_args(%arg7 = %cst, %arg8 = %cst_1, %arg9 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_5 = memref.subview %subview_2[0, %arg6, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg8, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg8, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg9, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg7, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_7 = memref.subview %subview_3[0, %arg6, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      vector.transfer_write %5, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<128x64xf32>
      %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 128, 64] : memref<128x64xf32> into memref<1x128x64xf32>
      %subview_4 = memref.subview %arg3[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      memref.copy %expand_shape, %subview_4 : memref<1x128x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %arg3 : memref<4x4096x64xf32>
  }
}


// -----// IR Dump After OneShotBufferizePass (one-shot-bufferize) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) -> memref<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_3 = memref.subview %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg6 = %c0 to %c4096 step %c64 iter_args(%arg7 = %cst, %arg8 = %cst_1, %arg9 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_5 = memref.subview %subview_2[0, %arg6, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg8, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg8, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg9, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg7, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_7 = memref.subview %subview_3[0, %arg6, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      vector.transfer_write %5, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<128x64xf32>
      %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 128, 64] : memref<128x64xf32> into memref<1x128x64xf32>
      %subview_4 = memref.subview %arg3[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      memref.copy %expand_shape, %subview_4 : memref<1x128x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %arg3 : memref<4x4096x64xf32>
  }
}


// -----// IR Dump After EmptyTensorEliminationPass (eliminate-empty-tensors) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) -> memref<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_3 = memref.subview %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg6 = %c0 to %c4096 step %c64 iter_args(%arg7 = %cst, %arg8 = %cst_1, %arg9 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_5 = memref.subview %subview_2[0, %arg6, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg8, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg8, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg9, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg7, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_7 = memref.subview %subview_3[0, %arg6, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      vector.transfer_write %5, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<128x64xf32>
      %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 128, 64] : memref<128x64xf32> into memref<1x128x64xf32>
      %subview_4 = memref.subview %arg3[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      memref.copy %expand_shape, %subview_4 : memref<1x128x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %arg3 : memref<4x4096x64xf32>
  }
}


// -----// IR Dump After EmptyTensorToAllocTensorPass (empty-tensor-to-alloc-tensor) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) -> memref<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_3 = memref.subview %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg6 = %c0 to %c4096 step %c64 iter_args(%arg7 = %cst, %arg8 = %cst_1, %arg9 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_5 = memref.subview %subview_2[0, %arg6, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg8, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg8, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg9, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg7, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_7 = memref.subview %subview_3[0, %arg6, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      vector.transfer_write %5, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<128x64xf32>
      %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 128, 64] : memref<128x64xf32> into memref<1x128x64xf32>
      %subview_4 = memref.subview %arg3[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      memref.copy %expand_shape, %subview_4 : memref<1x128x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %arg3 : memref<4x4096x64xf32>
  }
}


// -----// IR Dump After DropEquivalentBufferResultsPass (drop-equivalent-buffer-results) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) -> memref<4x4096x64xf32> attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg4, %arg5) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_3 = memref.subview %arg2[%arg4, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg6 = %c0 to %c4096 step %c64 iter_args(%arg7 = %cst, %arg8 = %cst_1, %arg9 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_5 = memref.subview %subview_2[0, %arg6, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg8, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg8, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg9, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg7, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_7 = memref.subview %subview_3[0, %arg6, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      vector.transfer_write %5, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<128x64xf32>
      %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 128, 64] : memref<128x64xf32> into memref<1x128x64xf32>
      %subview_4 = memref.subview %arg3[%arg4, %arg5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      memref.copy %expand_shape, %subview_4 : memref<1x128x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    return %arg3 : memref<4x4096x64xf32>
  }
}


// -----// IR Dump After BufferResultsToOutParamsPass (buffer-results-to-out-params) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg5, %arg6) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_2 = memref.subview %arg1[%arg5, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %subview_3 = memref.subview %arg2[%arg5, 0, 0] [1, 4096, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_5 = memref.subview %subview_2[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg9, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg9, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg10, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg8, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_7 = memref.subview %subview_3[0, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<1x4096x64xf16, strided<[262144, 64, 1], offset: ?>> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      vector.transfer_write %5, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<128x64xf32>
      %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 128, 64] : memref<128x64xf32> into memref<1x128x64xf32>
      %subview_4 = memref.subview %arg3[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      memref.copy %expand_shape, %subview_4 : memref<1x128x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
}


// -----// IR Dump After FoldMemRefAliasOpsPass (fold-memref-alias-ops) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg5, %arg6) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_3 = memref.subview %arg1[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg9, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg9, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg10, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg8, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_5 = memref.subview %arg2[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x64xf32>
      vector.transfer_write %5, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<128x64xf32>
      %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] output_shape [1, 128, 64] : memref<128x64xf32> into memref<1x128x64xf32>
      %subview_2 = memref.subview %arg3[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      memref.copy %expand_shape, %subview_2 : memref<1x128x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
}


// -----// IR Dump After RemoveAllocs (remove-allocs) //----- //
func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
  %0 = ub.poison : f16
  %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
  %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  scf.forall (%arg5, %arg6) = (0, 0) to (4, 4096) step (1, 128) {
    %subview = memref.subview %arg0[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
    %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
    %2:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      %subview_3 = memref.subview %arg1[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
      %6 = vector.transfer_read %collapse_shape_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
      %7 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
      %9 = arith.maximumf %arg9, %8 : vector<128xf32>
      %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
      %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
      %12 = arith.subf %7, %11 : vector<128x64xf32>
      %13 = math.exp %12 : vector<128x64xf32>
      %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
      %15 = arith.subf %arg9, %9 : vector<128xf32>
      %16 = math.exp %15 : vector<128xf32>
      %17 = arith.mulf %arg10, %16 : vector<128xf32>
      %18 = arith.addf %17, %14 : vector<128xf32>
      %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
      %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
      %21 = arith.mulf %arg8, %20 : vector<128x64xf32>
      %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
      %subview_5 = memref.subview %arg2[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
      %23 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
      %24 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    }
    %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
    %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
    %5 = arith.divf %2#0, %4 : vector<128x64xf32>
    %subview_2 = memref.subview %arg3[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    vector.transfer_write %5, %subview_2[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
  } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
  memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
  return
}

// -----// IR Dump After CSEPass (cse) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg5, %arg6) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_3 = memref.subview %arg1[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg9, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg9, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg10, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg8, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_5 = memref.subview %arg2[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %subview_2 = memref.subview %arg3[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      vector.transfer_write %5, %subview_2[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg5, %arg6) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_3 = memref.subview %arg1[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg9, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg9, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg10, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg8, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_5 = memref.subview %arg2[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %subview_2 = memref.subview %arg3[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      vector.transfer_write %5, %subview_2[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
}


// -----// IR Dump Bufferization //----- //
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg5, %arg6) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_3 = memref.subview %arg1[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg9, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg9, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg10, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg8, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_5 = memref.subview %arg2[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %subview_2 = memref.subview %arg3[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      vector.transfer_write %5, %subview_2[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
}
// -----// IR Dump After PrintIRPass (print-ir) //----- //
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg5, %arg6) = (0, 0) to (4, 4096) step (1, 128) {
      %subview = memref.subview %arg0[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %2:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_3 = memref.subview %arg1[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %6 = vector.transfer_read %collapse_shape_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %9 = arith.maximumf %arg9, %8 : vector<128xf32>
        %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
        %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
        %12 = arith.subf %7, %11 : vector<128x64xf32>
        %13 = math.exp %12 : vector<128x64xf32>
        %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %15 = arith.subf %arg9, %9 : vector<128xf32>
        %16 = math.exp %15 : vector<128xf32>
        %17 = arith.mulf %arg10, %16 : vector<128xf32>
        %18 = arith.addf %17, %14 : vector<128xf32>
        %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
        %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
        %21 = arith.mulf %arg8, %20 : vector<128x64xf32>
        %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
        %subview_5 = memref.subview %arg2[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %23 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %24 = vector.contract {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
      %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
      %5 = arith.divf %2#0, %4 : vector<128x64xf32>
      %subview_2 = memref.subview %arg3[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      vector.transfer_write %5, %subview_2[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    } {gc.kernel_name = "attention_f16_kernel", gc.tiling.stamp = 1 : i64}
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
}


// -----// IR Dump After GpuMapParallelLoopsPass (gpu-map-parallel-loops) //----- //
func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
  %0 = ub.poison : f16
  %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
  %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c4096_4 = arith.constant 4096 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  scf.parallel (%arg5, %arg6) = (%c0_2, %c0_3) to (%c4, %c4096_4) step (%c1, %c128) {
    %subview = memref.subview %arg0[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
    %1 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
    %2:3 = scf.for %arg7 = %c0 to %c4096 step %c64 iter_args(%arg8 = %cst, %arg9 = %cst_1, %arg10 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      %subview_6 = memref.subview %arg1[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape_7 = memref.collapse_shape %subview_6 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
      %6 = vector.transfer_read %collapse_shape_7[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
      %7 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %6, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %8 = vector.multi_reduction <maximumf>, %7, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
      %9 = arith.maximumf %arg9, %8 : vector<128xf32>
      %10 = vector.shape_cast %9 : vector<128xf32> to vector<128x1xf32>
      %11 = vector.broadcast %10 : vector<128x1xf32> to vector<128x64xf32>
      %12 = arith.subf %7, %11 : vector<128x64xf32>
      %13 = math.exp %12 : vector<128x64xf32>
      %14 = vector.multi_reduction <add>, %13, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
      %15 = arith.subf %arg9, %9 : vector<128xf32>
      %16 = math.exp %15 : vector<128xf32>
      %17 = arith.mulf %arg10, %16 : vector<128xf32>
      %18 = arith.addf %17, %14 : vector<128xf32>
      %19 = vector.shape_cast %16 : vector<128xf32> to vector<128x1xf32>
      %20 = vector.broadcast %19 : vector<128x1xf32> to vector<128x64xf32>
      %21 = arith.mulf %arg8, %20 : vector<128x64xf32>
      %22 = arith.truncf %13 : vector<128x64xf32> to vector<128x64xf16>
      %subview_8 = memref.subview %arg2[%arg5, %arg7, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape_9 = memref.collapse_shape %subview_8 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
      %23 = vector.transfer_read %collapse_shape_9[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
      %24 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %22, %23, %21 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      scf.yield %24, %9, %18 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    }
    %3 = vector.shape_cast %2#2 : vector<128xf32> to vector<128x1xf32>
    %4 = vector.broadcast %3 : vector<128x1xf32> to vector<128x64xf32>
    %5 = arith.divf %2#0, %4 : vector<128x64xf32>
    %subview_5 = memref.subview %arg3[%arg5, %arg6, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    vector.transfer_write %5, %subview_5[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    scf.reduce 
  } {gc.kernel_name = "attention_f16_kernel", mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>, #gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
  memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
  return
}

// -----// IR Dump After ConvertParallelLoopToGpuPass (convert-parallel-loops-to-gpu) //----- //
func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
  %0 = ub.poison : f16
  %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
  %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c4096_4 = arith.constant 4096 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c1_5 = arith.constant 1 : index
  %1 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%c4)[%c0_2, %c1]
  %2 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%c4096_4)[%c0_3, %c128]
  gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %1, %arg12 = %2, %arg13 = %c1_5) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1_5, %arg15 = %c1_5, %arg16 = %c1_5) {
    %3 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg5)[%c1, %c0_2]
    %4 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg6)[%c128, %c0_3]
    %subview = memref.subview %arg0[%3, %4, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
    %5 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
    %6:3 = scf.for %arg17 = %c0 to %c4096 step %c64 iter_args(%arg18 = %cst, %arg19 = %cst_1, %arg20 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      %subview_7 = memref.subview %arg1[%3, %arg17, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
      %10 = vector.transfer_read %collapse_shape_8[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
      %11 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %10, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %12 = vector.multi_reduction <maximumf>, %11, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
      %13 = arith.maximumf %arg19, %12 : vector<128xf32>
      %14 = vector.shape_cast %13 : vector<128xf32> to vector<128x1xf32>
      %15 = vector.broadcast %14 : vector<128x1xf32> to vector<128x64xf32>
      %16 = arith.subf %11, %15 : vector<128x64xf32>
      %17 = math.exp %16 : vector<128x64xf32>
      %18 = vector.multi_reduction <add>, %17, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
      %19 = arith.subf %arg19, %13 : vector<128xf32>
      %20 = math.exp %19 : vector<128xf32>
      %21 = arith.mulf %arg20, %20 : vector<128xf32>
      %22 = arith.addf %21, %18 : vector<128xf32>
      %23 = vector.shape_cast %20 : vector<128xf32> to vector<128x1xf32>
      %24 = vector.broadcast %23 : vector<128x1xf32> to vector<128x64xf32>
      %25 = arith.mulf %arg18, %24 : vector<128x64xf32>
      %26 = arith.truncf %17 : vector<128x64xf32> to vector<128x64xf16>
      %subview_9 = memref.subview %arg2[%3, %arg17, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape_10 = memref.collapse_shape %subview_9 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
      %27 = vector.transfer_read %collapse_shape_10[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
      %28 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %27, %25 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      scf.yield %28, %13, %22 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    }
    %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
    %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
    %9 = arith.divf %6#0, %8 : vector<128x64xf32>
    %subview_6 = memref.subview %arg3[%3, %4, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    vector.transfer_write %9, %subview_6[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    gpu.terminator
  } {SCFToGPU_visited, gc.kernel_name = "attention_f16_kernel"}
  memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
  return
}

// -----// IR Dump After GpuLaunchSinkIndexComputationsPass (gpu-launch-sink-index-computations) //----- //
func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
  %0 = ub.poison : f16
  %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
  %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c4096_4 = arith.constant 4096 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c1_5 = arith.constant 1 : index
  %1 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%c4)[%c0_2, %c1]
  %2 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%c4096_4)[%c0_3, %c128]
  gpu.launch blocks(%arg5, %arg6, %arg7) in (%arg11 = %1, %arg12 = %2, %arg13 = %c1_5) threads(%arg8, %arg9, %arg10) in (%arg14 = %c1_5, %arg15 = %c1_5, %arg16 = %c1_5) {
    %c1_6 = arith.constant 1 : index
    %c0_7 = arith.constant 0 : index
    %c128_8 = arith.constant 128 : index
    %c0_9 = arith.constant 0 : index
    %c0_10 = arith.constant 0 : index
    %3 = ub.poison : f16
    %cst_11 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_12 = arith.constant dense<0xFF800000> : vector<128xf32>
    %cst_13 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %c4096_14 = arith.constant 4096 : index
    %c64_15 = arith.constant 64 : index
    %4 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg5)[%c1_6, %c0_7]
    %5 = affine.apply affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>(%arg6)[%c128_8, %c0_9]
    %subview = memref.subview %arg0[%4, %5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
    %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
    %6 = vector.transfer_read %collapse_shape[%c0_10, %c0_10], %3 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
    %7:3 = scf.for %arg17 = %c0_10 to %c4096_14 step %c64_15 iter_args(%arg18 = %cst_11, %arg19 = %cst_12, %arg20 = %cst_13) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      %subview_17 = memref.subview %arg1[%4, %arg17, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape_18 = memref.collapse_shape %subview_17 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
      %11 = vector.transfer_read %collapse_shape_18[%c0_10, %c0_10], %3 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
      %12 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %6, %11, %cst_11 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      %13 = vector.multi_reduction <maximumf>, %12, %cst_12 [1] : vector<128x64xf32> to vector<128xf32>
      %14 = arith.maximumf %arg19, %13 : vector<128xf32>
      %15 = vector.shape_cast %14 : vector<128xf32> to vector<128x1xf32>
      %16 = vector.broadcast %15 : vector<128x1xf32> to vector<128x64xf32>
      %17 = arith.subf %12, %16 : vector<128x64xf32>
      %18 = math.exp %17 : vector<128x64xf32>
      %19 = vector.multi_reduction <add>, %18, %cst_13 [1] : vector<128x64xf32> to vector<128xf32>
      %20 = arith.subf %arg19, %14 : vector<128xf32>
      %21 = math.exp %20 : vector<128xf32>
      %22 = arith.mulf %arg20, %21 : vector<128xf32>
      %23 = arith.addf %22, %19 : vector<128xf32>
      %24 = vector.shape_cast %21 : vector<128xf32> to vector<128x1xf32>
      %25 = vector.broadcast %24 : vector<128x1xf32> to vector<128x64xf32>
      %26 = arith.mulf %arg18, %25 : vector<128x64xf32>
      %27 = arith.truncf %18 : vector<128x64xf32> to vector<128x64xf16>
      %subview_19 = memref.subview %arg2[%4, %arg17, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape_20 = memref.collapse_shape %subview_19 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
      %28 = vector.transfer_read %collapse_shape_20[%c0_10, %c0_10], %3 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
      %29 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %27, %28, %26 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
      scf.yield %29, %14, %23 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    }
    %8 = vector.shape_cast %7#2 : vector<128xf32> to vector<128x1xf32>
    %9 = vector.broadcast %8 : vector<128x1xf32> to vector<128x64xf32>
    %10 = arith.divf %7#0, %9 : vector<128x64xf32>
    %subview_16 = memref.subview %arg3[%4, %5, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    vector.transfer_write %10, %subview_16[%c0_10, %c0_10, %c0_10] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
    gpu.terminator
  } {SCFToGPU_visited, gc.kernel_name = "attention_f16_kernel"}
  memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
  return
}

// -----// IR Dump After GpuKernelOutliningPass (gpu-kernel-outlining) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}, gpu.container_module} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c4096_4 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c1_5 = arith.constant 1 : index
    %1 = affine.apply #map(%c4)[%c0_2, %c1]
    %2 = affine.apply #map(%c4096_4)[%c0_3, %c128]
    gpu.launch_func  @attention_f16_kernel::@attention_f16_kernel blocks in (%1, %2, %c1_5) threads in (%c1_5, %c1_5, %c1_5)  args(%arg0 : memref<4x4096x64xf16>, %arg1 : memref<4x4096x64xf16>, %arg2 : memref<4x4096x64xf16>, %arg3 : memref<4x4096x64xf32>)
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
  gpu.module @attention_f16_kernel {
    gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %0 = ub.poison : f16
      %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %cst_2 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_3 = arith.constant dense<0.000000e+00> : vector<128xf32>
      %c4096 = arith.constant 4096 : index
      %c64 = arith.constant 64 : index
      %1 = affine.apply #map1(%block_id_x)[%c1, %c0]
      %2 = affine.apply #map1(%block_id_y)[%c128, %c0_0]
      %subview = memref.subview %arg0[%1, %2, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %3 = vector.transfer_read %collapse_shape[%c0_1, %c0_1], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %4:3 = scf.for %arg4 = %c0_1 to %c4096 step %c64 iter_args(%arg5 = %cst, %arg6 = %cst_2, %arg7 = %cst_3) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_5 = memref.subview %arg1[%1, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %8 = vector.transfer_read %collapse_shape_6[%c0_1, %c0_1], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %9 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %8, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %10 = vector.multi_reduction <maximumf>, %9, %cst_2 [1] : vector<128x64xf32> to vector<128xf32>
        %11 = arith.maximumf %arg6, %10 : vector<128xf32>
        %12 = vector.shape_cast %11 : vector<128xf32> to vector<128x1xf32>
        %13 = vector.broadcast %12 : vector<128x1xf32> to vector<128x64xf32>
        %14 = arith.subf %9, %13 : vector<128x64xf32>
        %15 = math.exp %14 : vector<128x64xf32>
        %16 = vector.multi_reduction <add>, %15, %cst_3 [1] : vector<128x64xf32> to vector<128xf32>
        %17 = arith.subf %arg6, %11 : vector<128xf32>
        %18 = math.exp %17 : vector<128xf32>
        %19 = arith.mulf %arg7, %18 : vector<128xf32>
        %20 = arith.addf %19, %16 : vector<128xf32>
        %21 = vector.shape_cast %18 : vector<128xf32> to vector<128x1xf32>
        %22 = vector.broadcast %21 : vector<128x1xf32> to vector<128x64xf32>
        %23 = arith.mulf %arg5, %22 : vector<128x64xf32>
        %24 = arith.truncf %15 : vector<128x64xf32> to vector<128x64xf16>
        %subview_7 = memref.subview %arg2[%1, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %25 = vector.transfer_read %collapse_shape_8[%c0_1, %c0_1], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %26 = vector.contract {indexing_maps = [#map2, #map5, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %25, %23 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %26, %11, %20 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %5 = vector.shape_cast %4#2 : vector<128xf32> to vector<128x1xf32>
      %6 = vector.broadcast %5 : vector<128x1xf32> to vector<128x64xf32>
      %7 = arith.divf %4#0, %6 : vector<128x64xf32>
      %subview_4 = memref.subview %arg3[%1, %2, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      vector.transfer_write %7, %subview_4[%c0_1, %c0_1, %c0_1] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      gpu.return
    }
  }
}


// -----// IR Dump After GpuXeVMAttachTarget (xevm-attach-target) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}, gpu.container_module} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c4096_4 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c1_5 = arith.constant 1 : index
    %1 = affine.apply #map(%c4)[%c0_2, %c1]
    %2 = affine.apply #map(%c4096_4)[%c0_3, %c128]
    gpu.launch_func  @attention_f16_kernel::@attention_f16_kernel blocks in (%1, %2, %c1_5) threads in (%c1_5, %c1_5, %c1_5)  args(%arg0 : memref<4x4096x64xf16>, %arg1 : memref<4x4096x64xf16>, %arg2 : memref<4x4096x64xf16>, %arg3 : memref<4x4096x64xf32>)
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
  gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">] {
    gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %0 = ub.poison : f16
      %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %cst_2 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_3 = arith.constant dense<0.000000e+00> : vector<128xf32>
      %c4096 = arith.constant 4096 : index
      %c64 = arith.constant 64 : index
      %1 = affine.apply #map1(%block_id_x)[%c1, %c0]
      %2 = affine.apply #map1(%block_id_y)[%c128, %c0_0]
      %subview = memref.subview %arg0[%1, %2, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %3 = vector.transfer_read %collapse_shape[%c0_1, %c0_1], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %4:3 = scf.for %arg4 = %c0_1 to %c4096 step %c64 iter_args(%arg5 = %cst, %arg6 = %cst_2, %arg7 = %cst_3) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_5 = memref.subview %arg1[%1, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %8 = vector.transfer_read %collapse_shape_6[%c0_1, %c0_1], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %9 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %8, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %10 = vector.multi_reduction <maximumf>, %9, %cst_2 [1] : vector<128x64xf32> to vector<128xf32>
        %11 = arith.maximumf %arg6, %10 : vector<128xf32>
        %12 = vector.shape_cast %11 : vector<128xf32> to vector<128x1xf32>
        %13 = vector.broadcast %12 : vector<128x1xf32> to vector<128x64xf32>
        %14 = arith.subf %9, %13 : vector<128x64xf32>
        %15 = math.exp %14 : vector<128x64xf32>
        %16 = vector.multi_reduction <add>, %15, %cst_3 [1] : vector<128x64xf32> to vector<128xf32>
        %17 = arith.subf %arg6, %11 : vector<128xf32>
        %18 = math.exp %17 : vector<128xf32>
        %19 = arith.mulf %arg7, %18 : vector<128xf32>
        %20 = arith.addf %19, %16 : vector<128xf32>
        %21 = vector.shape_cast %18 : vector<128xf32> to vector<128x1xf32>
        %22 = vector.broadcast %21 : vector<128x1xf32> to vector<128x64xf32>
        %23 = arith.mulf %arg5, %22 : vector<128x64xf32>
        %24 = arith.truncf %15 : vector<128x64xf32> to vector<128x64xf16>
        %subview_7 = memref.subview %arg2[%1, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %25 = vector.transfer_read %collapse_shape_8[%c0_1, %c0_1], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %26 = vector.contract {indexing_maps = [#map2, #map5, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %25, %23 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %26, %11, %20 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %5 = vector.shape_cast %4#2 : vector<128xf32> to vector<128x1xf32>
      %6 = vector.broadcast %5 : vector<128x1xf32> to vector<128x64xf32>
      %7 = arith.divf %4#0, %6 : vector<128x64xf32>
      %subview_4 = memref.subview %arg3[%1, %2, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      vector.transfer_write %7, %subview_4[%c0_1, %c0_1, %c0_1] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      gpu.return
    }
  }
}


// -----// IR Dump After GpuKernelOutline (gpu-kernel-outline) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}, gpu.container_module} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}) attributes {gc.num_kernels = 1 : i32} {
    %0 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c0_2 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c4096_4 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c1_5 = arith.constant 1 : index
    %1 = affine.apply #map(%c4)[%c0_2, %c1]
    %2 = affine.apply #map(%c4096_4)[%c0_3, %c128]
    gpu.launch_func  @attention_f16_kernel::@attention_f16_kernel blocks in (%1, %2, %c1_5) threads in (%c1_5, %c1_5, %c1_5)  args(%arg0 : memref<4x4096x64xf16>, %arg1 : memref<4x4096x64xf16>, %arg2 : memref<4x4096x64xf16>, %arg3 : memref<4x4096x64xf32>)
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
  gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">] {
    gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c0_0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %0 = ub.poison : f16
      %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %cst_2 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_3 = arith.constant dense<0.000000e+00> : vector<128xf32>
      %c4096 = arith.constant 4096 : index
      %c64 = arith.constant 64 : index
      %1 = affine.apply #map1(%block_id_x)[%c1, %c0]
      %2 = affine.apply #map1(%block_id_y)[%c128, %c0_0]
      %subview = memref.subview %arg0[%1, %2, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %3 = vector.transfer_read %collapse_shape[%c0_1, %c0_1], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %4:3 = scf.for %arg4 = %c0_1 to %c4096 step %c64 iter_args(%arg5 = %cst, %arg6 = %cst_2, %arg7 = %cst_3) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_5 = memref.subview %arg1[%1, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %8 = vector.transfer_read %collapse_shape_6[%c0_1, %c0_1], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %9 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %8, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %10 = vector.multi_reduction <maximumf>, %9, %cst_2 [1] : vector<128x64xf32> to vector<128xf32>
        %11 = arith.maximumf %arg6, %10 : vector<128xf32>
        %12 = vector.shape_cast %11 : vector<128xf32> to vector<128x1xf32>
        %13 = vector.broadcast %12 : vector<128x1xf32> to vector<128x64xf32>
        %14 = arith.subf %9, %13 : vector<128x64xf32>
        %15 = math.exp %14 : vector<128x64xf32>
        %16 = vector.multi_reduction <add>, %15, %cst_3 [1] : vector<128x64xf32> to vector<128xf32>
        %17 = arith.subf %arg6, %11 : vector<128xf32>
        %18 = math.exp %17 : vector<128xf32>
        %19 = arith.mulf %arg7, %18 : vector<128xf32>
        %20 = arith.addf %19, %16 : vector<128xf32>
        %21 = vector.shape_cast %18 : vector<128xf32> to vector<128x1xf32>
        %22 = vector.broadcast %21 : vector<128x1xf32> to vector<128x64xf32>
        %23 = arith.mulf %arg5, %22 : vector<128x64xf32>
        %24 = arith.truncf %15 : vector<128x64xf32> to vector<128x64xf16>
        %subview_7 = memref.subview %arg2[%1, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_8 = memref.collapse_shape %subview_7 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %25 = vector.transfer_read %collapse_shape_8[%c0_1, %c0_1], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %26 = vector.contract {indexing_maps = [#map2, #map5, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %25, %23 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %26, %11, %20 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %5 = vector.shape_cast %4#2 : vector<128xf32> to vector<128x1xf32>
      %6 = vector.broadcast %5 : vector<128x1xf32> to vector<128x64xf32>
      %7 = arith.divf %4#0, %6 : vector<128x64xf32>
      %subview_4 = memref.subview %arg3[%1, %2, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      vector.transfer_write %7, %subview_4[%c0_1, %c0_1, %c0_1] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      gpu.return
    }
  }
}


// -----// IR Dump After AddContextArg (add-ctx-arg) //----- //
func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}, %arg5: memref<i8>) attributes {gc.num_kernels = 1 : i32} {
  %0 = ub.poison : f16
  %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<128xf32>
  %cst_1 = arith.constant dense<0xFF800000> : vector<128xf32>
  %c4096 = arith.constant 4096 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c0_2 = arith.constant 0 : index
  %c0_3 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c4096_4 = arith.constant 4096 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %c1_5 = arith.constant 1 : index
  %1 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%c4)[%c0_2, %c1]
  %2 = affine.apply affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>(%c4096_4)[%c0_3, %c128]
  gpu.launch_func  @attention_f16_kernel::@attention_f16_kernel blocks in (%1, %2, %c1_5) threads in (%c1_5, %c1_5, %c1_5)  args(%arg0 : memref<4x4096x64xf16>, %arg1 : memref<4x4096x64xf16>, %arg2 : memref<4x4096x64xf16>, %arg3 : memref<4x4096x64xf32>)
  memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
  return
}

// -----// IR Dump After CSEPass (cse) //----- //
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d1)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}, gpu.container_module} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}, %arg5: memref<i8>) attributes {gc.num_kernels = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %0 = affine.apply #map(%c4)[%c0, %c1]
    %1 = affine.apply #map(%c4096)[%c0, %c128]
    gpu.launch_func  @attention_f16_kernel::@attention_f16_kernel blocks in (%0, %1, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4x4096x64xf16>, %arg1 : memref<4x4096x64xf16>, %arg2 : memref<4x4096x64xf16>, %arg3 : memref<4x4096x64xf32>)
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
  gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">] {
    gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %0 = ub.poison : f16
      %cst = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128xf32>
      %c4096 = arith.constant 4096 : index
      %c64 = arith.constant 64 : index
      %1 = affine.apply #map1(%block_id_x)[%c1, %c0]
      %2 = affine.apply #map1(%block_id_y)[%c128, %c0]
      %subview = memref.subview %arg0[%1, %2, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %3 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %4:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst, %arg6 = %cst_0, %arg7 = %cst_1) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_3 = memref.subview %arg1[%1, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %8 = vector.transfer_read %collapse_shape_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %9 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %8, %cst : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        %10 = vector.multi_reduction <maximumf>, %9, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %11 = arith.maximumf %arg6, %10 : vector<128xf32>
        %12 = vector.shape_cast %11 : vector<128xf32> to vector<128x1xf32>
        %13 = vector.broadcast %12 : vector<128x1xf32> to vector<128x64xf32>
        %14 = arith.subf %9, %13 : vector<128x64xf32>
        %15 = math.exp %14 : vector<128x64xf32>
        %16 = vector.multi_reduction <add>, %15, %cst_1 [1] : vector<128x64xf32> to vector<128xf32>
        %17 = arith.subf %arg6, %11 : vector<128xf32>
        %18 = math.exp %17 : vector<128xf32>
        %19 = arith.mulf %arg7, %18 : vector<128xf32>
        %20 = arith.addf %19, %16 : vector<128xf32>
        %21 = vector.shape_cast %18 : vector<128xf32> to vector<128x1xf32>
        %22 = vector.broadcast %21 : vector<128x1xf32> to vector<128x64xf32>
        %23 = arith.mulf %arg5, %22 : vector<128x64xf32>
        %24 = arith.truncf %15 : vector<128x64xf32> to vector<128x64xf16>
        %subview_5 = memref.subview %arg2[%1, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_6 = memref.collapse_shape %subview_5 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %25 = vector.transfer_read %collapse_shape_6[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %26 = vector.contract {indexing_maps = [#map2, #map5, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %25, %23 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
        scf.yield %26, %11, %20 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %5 = vector.shape_cast %4#2 : vector<128xf32> to vector<128x1xf32>
      %6 = vector.broadcast %5 : vector<128x1xf32> to vector<128x64xf32>
      %7 = arith.divf %4#0, %6 : vector<128x64xf32>
      %subview_2 = memref.subview %arg3[%1, %2, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      vector.transfer_write %7, %subview_2[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      gpu.return
    }
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
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


// -----// IR Dump KernelOutlining //----- //
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
      %1 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%block_id_y]
      %subview = memref.subview %arg0[%block_id_x, %1, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %2 = vector.transfer_read %collapse_shape[%c0, %c0], %0 {in_bounds = [true, true]} : memref<128x64xf16, strided<[64, 1], offset: ?>>, vector<128x64xf16>
      %3:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_3 = memref.subview %arg1[%block_id_x, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_4 = memref.collapse_shape %subview_3 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %7 = vector.transfer_read %collapse_shape_4[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xf16, strided<[64, 1], offset: ?>>, vector<64x64xf16>
        %8 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %2, %7, %cst_1 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
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
        %25 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %23, %24, %22 : vector<128x64xf16>, vector<64x64xf16> into vector<128x64xf32>
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
// -----// IR Dump After PrintIRPass (print-ir) //----- //
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


// -----// IR Dump After ConvertVectorToXeGPU (convert-vector-to-xegpu) //----- //
#map = affine_map<()[s0] -> (s0 * 128)>
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
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %0 = affine.apply #map()[%block_id_y]
      %subview = memref.subview %arg0[%block_id_x, %0, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>>
      %collapse_shape = memref.collapse_shape %subview [[0, 1], [2]] : memref<1x128x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<128x64xf16, strided<[64, 1], offset: ?>>
      %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %collapse_shape : memref<128x64xf16, strided<[64, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %offset, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %6:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %subview_9 = memref.subview %arg1[%block_id_x, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_10 = memref.collapse_shape %subview_9 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %base_buffer_11, %offset_12, %sizes_13:2, %strides_14:2 = memref.extract_strided_metadata %collapse_shape_10 : memref<64x64xf16, strided<[64, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
        %intptr_15 = memref.extract_aligned_pointer_as_index %base_buffer_11 : memref<f16> -> index
        %14 = arith.muli %offset_12, %c2 : index
        %15 = arith.addi %intptr_15, %14 : index
        %16 = arith.index_cast %15 : index to i64
        %17 = xegpu.create_nd_tdesc %16, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %18 = xegpu.load_nd %17[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %19 = vector.transpose %18, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %20 = xegpu.dpas %5, %19, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %21 = vector.multi_reduction <maximumf>, %20, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %22 = arith.maximumf %arg6, %21 : vector<128xf32>
        %23 = vector.shape_cast %22 : vector<128xf32> to vector<128x1xf32>
        %24 = vector.broadcast %23 : vector<128x1xf32> to vector<128x64xf32>
        %25 = arith.subf %20, %24 : vector<128x64xf32>
        %26 = math.exp %25 : vector<128x64xf32>
        %27 = vector.multi_reduction <add>, %26, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %28 = arith.subf %arg6, %22 : vector<128xf32>
        %29 = math.exp %28 : vector<128xf32>
        %30 = arith.mulf %arg7, %29 : vector<128xf32>
        %31 = arith.addf %30, %27 : vector<128xf32>
        %32 = vector.shape_cast %29 : vector<128xf32> to vector<128x1xf32>
        %33 = vector.broadcast %32 : vector<128x1xf32> to vector<128x64xf32>
        %34 = arith.mulf %arg5, %33 : vector<128x64xf32>
        %35 = arith.truncf %26 : vector<128x64xf32> to vector<128x64xf16>
        %subview_16 = memref.subview %arg2[%block_id_x, %arg4, 0] [1, 64, 64] [1, 1, 1] : memref<4x4096x64xf16> to memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>>
        %collapse_shape_17 = memref.collapse_shape %subview_16 [[0, 1], [2]] : memref<1x64x64xf16, strided<[262144, 64, 1], offset: ?>> into memref<64x64xf16, strided<[64, 1], offset: ?>>
        %base_buffer_18, %offset_19, %sizes_20:2, %strides_21:2 = memref.extract_strided_metadata %collapse_shape_17 : memref<64x64xf16, strided<[64, 1], offset: ?>> -> memref<f16>, index, index, index, index, index
        %intptr_22 = memref.extract_aligned_pointer_as_index %base_buffer_18 : memref<f16> -> index
        %36 = arith.muli %offset_19, %c2 : index
        %37 = arith.addi %intptr_22, %36 : index
        %38 = arith.index_cast %37 : index to i64
        %39 = xegpu.create_nd_tdesc %38, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %40 = xegpu.load_nd %39[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %41 = xegpu.dpas %35, %40, %34 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %41, %22, %31 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
      %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
      %9 = arith.divf %6#0, %8 : vector<128x64xf32>
      %subview_2 = memref.subview %arg3[%block_id_x, %0, 0] [1, 128, 64] [1, 1, 1] : memref<4x4096x64xf32> to memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>>
      %subview_3 = memref.subview %subview_2[0, 0, 0] [1, 128, 64] [1, 1, 1] : memref<1x128x64xf32, strided<[262144, 64, 1], offset: ?>> to memref<128x64xf32, strided<[64, 1], offset: ?>>
      %base_buffer_4, %offset_5, %sizes_6:2, %strides_7:2 = memref.extract_strided_metadata %subview_3 : memref<128x64xf32, strided<[64, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
      %intptr_8 = memref.extract_aligned_pointer_as_index %base_buffer_4 : memref<f32> -> index
      %10 = arith.muli %offset_5, %c4 : index
      %11 = arith.addi %intptr_8, %10 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = xegpu.create_nd_tdesc %12, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %9, %13[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}


// -----// IR Dump After ExpandStridedMetadataPass (expand-strided-metadata) //----- //
#map = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 64)>
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
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %0 = affine.apply #map()[%block_id_x, %block_id_y]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %0, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %6:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %15 = affine.apply #map1()[%block_id_x, %arg4]
        %intptr_11 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
        %16 = arith.muli %15, %c2 : index
        %17 = arith.addi %intptr_11, %16 : index
        %18 = arith.index_cast %17 : index to i64
        %19 = xegpu.create_nd_tdesc %18, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %20 = xegpu.load_nd %19[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %21 = vector.transpose %20, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %22 = xegpu.dpas %5, %21, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %23 = vector.multi_reduction <maximumf>, %22, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %24 = arith.maximumf %arg6, %23 : vector<128xf32>
        %25 = vector.shape_cast %24 : vector<128xf32> to vector<128x1xf32>
        %26 = vector.broadcast %25 : vector<128x1xf32> to vector<128x64xf32>
        %27 = arith.subf %22, %26 : vector<128x64xf32>
        %28 = math.exp %27 : vector<128x64xf32>
        %29 = vector.multi_reduction <add>, %28, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %30 = arith.subf %arg6, %24 : vector<128xf32>
        %31 = math.exp %30 : vector<128xf32>
        %32 = arith.mulf %arg7, %31 : vector<128xf32>
        %33 = arith.addf %32, %29 : vector<128xf32>
        %34 = vector.shape_cast %31 : vector<128xf32> to vector<128x1xf32>
        %35 = vector.broadcast %34 : vector<128x1xf32> to vector<128x64xf32>
        %36 = arith.mulf %arg5, %35 : vector<128x64xf32>
        %37 = arith.truncf %28 : vector<128x64xf32> to vector<128x64xf16>
        %base_buffer_12, %offset_13, %sizes_14:3, %strides_15:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %38 = affine.apply #map1()[%block_id_x, %arg4]
        %intptr_16 = memref.extract_aligned_pointer_as_index %base_buffer_12 : memref<f16> -> index
        %39 = arith.muli %38, %c2 : index
        %40 = arith.addi %intptr_16, %39 : index
        %41 = arith.index_cast %40 : index to i64
        %42 = xegpu.create_nd_tdesc %41, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %43 = xegpu.load_nd %42[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %44 = xegpu.dpas %37, %43, %36 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %44, %24, %33 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
      %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
      %9 = arith.divf %6#0, %8 : vector<128x64xf32>
      %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
      %10 = affine.apply #map()[%block_id_x, %block_id_y]
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %11 = arith.muli %10, %c4 : index
      %12 = arith.addi %intptr_6, %11 : index
      %13 = arith.index_cast %12 : index to i64
      %14 = xegpu.create_nd_tdesc %13, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %9, %14[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
#map = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 64)>
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
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %0 = affine.apply #map()[%block_id_x, %block_id_y]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %0, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %6:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %14 = affine.apply #map1()[%block_id_x, %arg4]
        %intptr_11 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
        %15 = arith.muli %14, %c2 : index
        %16 = arith.addi %intptr_11, %15 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = xegpu.create_nd_tdesc %17, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %19 = xegpu.load_nd %18[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %20 = vector.transpose %19, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %21 = xegpu.dpas %5, %20, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %22 = vector.multi_reduction <maximumf>, %21, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %23 = arith.maximumf %arg6, %22 : vector<128xf32>
        %24 = vector.shape_cast %23 : vector<128xf32> to vector<128x1xf32>
        %25 = vector.broadcast %24 : vector<128x1xf32> to vector<128x64xf32>
        %26 = arith.subf %21, %25 : vector<128x64xf32>
        %27 = math.exp %26 : vector<128x64xf32>
        %28 = vector.multi_reduction <add>, %27, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %29 = arith.subf %arg6, %23 : vector<128xf32>
        %30 = math.exp %29 : vector<128xf32>
        %31 = arith.mulf %arg7, %30 : vector<128xf32>
        %32 = arith.addf %31, %28 : vector<128xf32>
        %33 = vector.shape_cast %30 : vector<128xf32> to vector<128x1xf32>
        %34 = vector.broadcast %33 : vector<128x1xf32> to vector<128x64xf32>
        %35 = arith.mulf %arg5, %34 : vector<128x64xf32>
        %36 = arith.truncf %27 : vector<128x64xf32> to vector<128x64xf16>
        %base_buffer_12, %offset_13, %sizes_14:3, %strides_15:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %intptr_16 = memref.extract_aligned_pointer_as_index %base_buffer_12 : memref<f16> -> index
        %37 = arith.addi %intptr_16, %15 : index
        %38 = arith.index_cast %37 : index to i64
        %39 = xegpu.create_nd_tdesc %38, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %40 = xegpu.load_nd %39[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %41 = xegpu.dpas %36, %40, %35 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %41, %23, %32 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
      %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
      %9 = arith.divf %6#0, %8 : vector<128x64xf32>
      %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %10 = arith.muli %0, %c4 : index
      %11 = arith.addi %intptr_6, %10 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = xegpu.create_nd_tdesc %12, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %9, %13[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}


// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
#map = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 64)>
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
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %0 = affine.apply #map()[%block_id_x, %block_id_y]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %0, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %6:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %14 = affine.apply #map1()[%block_id_x, %arg4]
        %intptr_11 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
        %15 = arith.muli %14, %c2 : index
        %16 = arith.addi %intptr_11, %15 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = xegpu.create_nd_tdesc %17, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %19 = xegpu.load_nd %18[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %20 = vector.transpose %19, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %21 = xegpu.dpas %5, %20, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %22 = vector.multi_reduction <maximumf>, %21, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %23 = arith.maximumf %arg6, %22 : vector<128xf32>
        %24 = vector.shape_cast %23 : vector<128xf32> to vector<128x1xf32>
        %25 = vector.broadcast %24 : vector<128x1xf32> to vector<128x64xf32>
        %26 = arith.subf %21, %25 : vector<128x64xf32>
        %27 = math.exp %26 : vector<128x64xf32>
        %28 = vector.multi_reduction <add>, %27, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %29 = arith.subf %arg6, %23 : vector<128xf32>
        %30 = math.exp %29 : vector<128xf32>
        %31 = arith.mulf %arg7, %30 : vector<128xf32>
        %32 = arith.addf %31, %28 : vector<128xf32>
        %33 = vector.shape_cast %30 : vector<128xf32> to vector<128x1xf32>
        %34 = vector.broadcast %33 : vector<128x1xf32> to vector<128x64xf32>
        %35 = arith.mulf %arg5, %34 : vector<128x64xf32>
        %36 = arith.truncf %27 : vector<128x64xf32> to vector<128x64xf16>
        %base_buffer_12, %offset_13, %sizes_14:3, %strides_15:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %intptr_16 = memref.extract_aligned_pointer_as_index %base_buffer_12 : memref<f16> -> index
        %37 = arith.addi %intptr_16, %15 : index
        %38 = arith.index_cast %37 : index to i64
        %39 = xegpu.create_nd_tdesc %38, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %40 = xegpu.load_nd %39[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %41 = xegpu.dpas %36, %40, %35 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %41, %23, %32 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
      %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
      %9 = arith.divf %6#0, %8 : vector<128x64xf32>
      %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %10 = arith.muli %0, %c4 : index
      %11 = arith.addi %intptr_6, %10 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = xegpu.create_nd_tdesc %12, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %9, %13[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}


// -----// IR Dump VectorToXegpu //----- //
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
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %0 = affine.apply affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 8192)>()[%block_id_x, %block_id_y]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %0, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %6:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %14 = affine.apply affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 64)>()[%block_id_x, %arg4]
        %intptr_11 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
        %15 = arith.muli %14, %c2 : index
        %16 = arith.addi %intptr_11, %15 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = xegpu.create_nd_tdesc %17, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %19 = xegpu.load_nd %18[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %20 = vector.transpose %19, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %21 = xegpu.dpas %5, %20, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %22 = vector.multi_reduction <maximumf>, %21, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %23 = arith.maximumf %arg6, %22 : vector<128xf32>
        %24 = vector.shape_cast %23 : vector<128xf32> to vector<128x1xf32>
        %25 = vector.broadcast %24 : vector<128x1xf32> to vector<128x64xf32>
        %26 = arith.subf %21, %25 : vector<128x64xf32>
        %27 = math.exp %26 : vector<128x64xf32>
        %28 = vector.multi_reduction <add>, %27, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %29 = arith.subf %arg6, %23 : vector<128xf32>
        %30 = math.exp %29 : vector<128xf32>
        %31 = arith.mulf %arg7, %30 : vector<128xf32>
        %32 = arith.addf %31, %28 : vector<128xf32>
        %33 = vector.shape_cast %30 : vector<128xf32> to vector<128x1xf32>
        %34 = vector.broadcast %33 : vector<128x1xf32> to vector<128x64xf32>
        %35 = arith.mulf %arg5, %34 : vector<128x64xf32>
        %36 = arith.truncf %27 : vector<128x64xf32> to vector<128x64xf16>
        %base_buffer_12, %offset_13, %sizes_14:3, %strides_15:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %intptr_16 = memref.extract_aligned_pointer_as_index %base_buffer_12 : memref<f16> -> index
        %37 = arith.addi %intptr_16, %15 : index
        %38 = arith.index_cast %37 : index to i64
        %39 = xegpu.create_nd_tdesc %38, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %40 = xegpu.load_nd %39[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %41 = xegpu.dpas %36, %40, %35 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %41, %23, %32 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
      %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
      %9 = arith.divf %6#0, %8 : vector<128x64xf32>
      %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %10 = arith.muli %0, %c4 : index
      %11 = arith.addi %intptr_6, %10 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = xegpu.create_nd_tdesc %12, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %9, %13[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}
// -----// IR Dump After PrintIRPass (print-ir) //----- //
#map = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 64)>
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
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %0 = affine.apply #map()[%block_id_x, %block_id_y]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %0, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %6:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %14 = affine.apply #map1()[%block_id_x, %arg4]
        %intptr_11 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
        %15 = arith.muli %14, %c2 : index
        %16 = arith.addi %intptr_11, %15 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = xegpu.create_nd_tdesc %17, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %19 = xegpu.load_nd %18[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %20 = vector.transpose %19, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %21 = xegpu.dpas %5, %20, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %22 = vector.multi_reduction <maximumf>, %21, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %23 = arith.maximumf %arg6, %22 : vector<128xf32>
        %24 = vector.shape_cast %23 : vector<128xf32> to vector<128x1xf32>
        %25 = vector.broadcast %24 : vector<128x1xf32> to vector<128x64xf32>
        %26 = arith.subf %21, %25 : vector<128x64xf32>
        %27 = math.exp %26 : vector<128x64xf32>
        %28 = vector.multi_reduction <add>, %27, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %29 = arith.subf %arg6, %23 : vector<128xf32>
        %30 = math.exp %29 : vector<128xf32>
        %31 = arith.mulf %arg7, %30 : vector<128xf32>
        %32 = arith.addf %31, %28 : vector<128xf32>
        %33 = vector.shape_cast %30 : vector<128xf32> to vector<128x1xf32>
        %34 = vector.broadcast %33 : vector<128x1xf32> to vector<128x64xf32>
        %35 = arith.mulf %arg5, %34 : vector<128x64xf32>
        %36 = arith.truncf %27 : vector<128x64xf32> to vector<128x64xf16>
        %base_buffer_12, %offset_13, %sizes_14:3, %strides_15:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %intptr_16 = memref.extract_aligned_pointer_as_index %base_buffer_12 : memref<f16> -> index
        %37 = arith.addi %intptr_16, %15 : index
        %38 = arith.index_cast %37 : index to i64
        %39 = xegpu.create_nd_tdesc %38, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %40 = xegpu.load_nd %39[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %41 = xegpu.dpas %36, %40, %35 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %41, %23, %32 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
      %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
      %9 = arith.divf %6#0, %8 : vector<128x64xf32>
      %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %10 = arith.muli %0, %c4 : index
      %11 = arith.addi %intptr_6, %10 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = xegpu.create_nd_tdesc %12, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %9, %13[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}


// -----// IR Dump After CSEPass (cse) //----- //
#map = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 64)>
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
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %0 = affine.apply #map()[%block_id_x, %block_id_y]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %0, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %6:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %14 = affine.apply #map1()[%block_id_x, %arg4]
        %intptr_11 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
        %15 = arith.muli %14, %c2 : index
        %16 = arith.addi %intptr_11, %15 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = xegpu.create_nd_tdesc %17, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %19 = xegpu.load_nd %18[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %20 = vector.transpose %19, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %21 = xegpu.dpas %5, %20, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %22 = vector.multi_reduction <maximumf>, %21, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %23 = arith.maximumf %arg6, %22 : vector<128xf32>
        %24 = vector.shape_cast %23 : vector<128xf32> to vector<128x1xf32>
        %25 = vector.broadcast %24 : vector<128x1xf32> to vector<128x64xf32>
        %26 = arith.subf %21, %25 : vector<128x64xf32>
        %27 = math.exp %26 : vector<128x64xf32>
        %28 = vector.multi_reduction <add>, %27, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %29 = arith.subf %arg6, %23 : vector<128xf32>
        %30 = math.exp %29 : vector<128xf32>
        %31 = arith.mulf %arg7, %30 : vector<128xf32>
        %32 = arith.addf %31, %28 : vector<128xf32>
        %33 = vector.shape_cast %30 : vector<128xf32> to vector<128x1xf32>
        %34 = vector.broadcast %33 : vector<128x1xf32> to vector<128x64xf32>
        %35 = arith.mulf %arg5, %34 : vector<128x64xf32>
        %36 = arith.truncf %27 : vector<128x64xf32> to vector<128x64xf16>
        %base_buffer_12, %offset_13, %sizes_14:3, %strides_15:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %intptr_16 = memref.extract_aligned_pointer_as_index %base_buffer_12 : memref<f16> -> index
        %37 = arith.addi %intptr_16, %15 : index
        %38 = arith.index_cast %37 : index to i64
        %39 = xegpu.create_nd_tdesc %38, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %40 = xegpu.load_nd %39[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %41 = xegpu.dpas %36, %40, %35 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %41, %23, %32 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
      %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
      %9 = arith.divf %6#0, %8 : vector<128x64xf32>
      %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %10 = arith.muli %0, %c4 : index
      %11 = arith.addi %intptr_6, %10 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = xegpu.create_nd_tdesc %12, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %9, %13[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}


// -----// IR Dump After ConvertVectorToSCF (convert-vector-to-scf) //----- //
#map = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 64)>
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
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %0 = affine.apply #map()[%block_id_x, %block_id_y]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %0, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %6:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %14 = affine.apply #map1()[%block_id_x, %arg4]
        %intptr_11 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
        %15 = arith.muli %14, %c2 : index
        %16 = arith.addi %intptr_11, %15 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = xegpu.create_nd_tdesc %17, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %19 = xegpu.load_nd %18[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %20 = vector.transpose %19, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %21 = xegpu.dpas %5, %20, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %22 = vector.multi_reduction <maximumf>, %21, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %23 = arith.maximumf %arg6, %22 : vector<128xf32>
        %24 = vector.shape_cast %23 : vector<128xf32> to vector<128x1xf32>
        %25 = vector.broadcast %24 : vector<128x1xf32> to vector<128x64xf32>
        %26 = arith.subf %21, %25 : vector<128x64xf32>
        %27 = math.exp %26 : vector<128x64xf32>
        %28 = vector.multi_reduction <add>, %27, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %29 = arith.subf %arg6, %23 : vector<128xf32>
        %30 = math.exp %29 : vector<128xf32>
        %31 = arith.mulf %arg7, %30 : vector<128xf32>
        %32 = arith.addf %31, %28 : vector<128xf32>
        %33 = vector.shape_cast %30 : vector<128xf32> to vector<128x1xf32>
        %34 = vector.broadcast %33 : vector<128x1xf32> to vector<128x64xf32>
        %35 = arith.mulf %arg5, %34 : vector<128x64xf32>
        %36 = arith.truncf %27 : vector<128x64xf32> to vector<128x64xf16>
        %base_buffer_12, %offset_13, %sizes_14:3, %strides_15:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %intptr_16 = memref.extract_aligned_pointer_as_index %base_buffer_12 : memref<f16> -> index
        %37 = arith.addi %intptr_16, %15 : index
        %38 = arith.index_cast %37 : index to i64
        %39 = xegpu.create_nd_tdesc %38, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %40 = xegpu.load_nd %39[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %41 = xegpu.dpas %36, %40, %35 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %41, %23, %32 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
      %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
      %9 = arith.divf %6#0, %8 : vector<128x64xf32>
      %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %10 = arith.muli %0, %c4 : index
      %11 = arith.addi %intptr_6, %10 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = xegpu.create_nd_tdesc %12, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %9, %13[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}


// -----// IR Dump After GpuXeVMAttachTarget (xevm-attach-target) //----- //
#map = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 8192)>
#map1 = affine_map<()[s0, s1] -> (s0 * 262144 + s1 * 64)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}, gpu.container_module} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}, %arg5: memref<i8>) attributes {gc.num_kernels = 1 : i32} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    gpu.launch_func  @attention_f16_kernel::@attention_f16_kernel blocks in (%c4, %c32, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4x4096x64xf16>, %arg1 : memref<4x4096x64xf16>, %arg2 : memref<4x4096x64xf16>, %arg3 : memref<4x4096x64xf32>)
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
  gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">, #xevm.target<O = 3>] {
    gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %0 = affine.apply #map()[%block_id_x, %block_id_y]
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %1 = arith.muli %0, %c2 : index
      %2 = arith.addi %intptr, %1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = xegpu.create_nd_tdesc %3, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %5 = xegpu.load_nd %4[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %6:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %14 = affine.apply #map1()[%block_id_x, %arg4]
        %intptr_11 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
        %15 = arith.muli %14, %c2 : index
        %16 = arith.addi %intptr_11, %15 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = xegpu.create_nd_tdesc %17, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %19 = xegpu.load_nd %18[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %20 = vector.transpose %19, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %21 = xegpu.dpas %5, %20, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %22 = vector.multi_reduction <maximumf>, %21, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %23 = arith.maximumf %arg6, %22 : vector<128xf32>
        %24 = vector.shape_cast %23 : vector<128xf32> to vector<128x1xf32>
        %25 = vector.broadcast %24 : vector<128x1xf32> to vector<128x64xf32>
        %26 = arith.subf %21, %25 : vector<128x64xf32>
        %27 = math.exp %26 : vector<128x64xf32>
        %28 = vector.multi_reduction <add>, %27, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %29 = arith.subf %arg6, %23 : vector<128xf32>
        %30 = math.exp %29 : vector<128xf32>
        %31 = arith.mulf %arg7, %30 : vector<128xf32>
        %32 = arith.addf %31, %28 : vector<128xf32>
        %33 = vector.shape_cast %30 : vector<128xf32> to vector<128x1xf32>
        %34 = vector.broadcast %33 : vector<128x1xf32> to vector<128x64xf32>
        %35 = arith.mulf %arg5, %34 : vector<128x64xf32>
        %36 = arith.truncf %27 : vector<128x64xf32> to vector<128x64xf16>
        %base_buffer_12, %offset_13, %sizes_14:3, %strides_15:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %intptr_16 = memref.extract_aligned_pointer_as_index %base_buffer_12 : memref<f16> -> index
        %37 = arith.addi %intptr_16, %15 : index
        %38 = arith.index_cast %37 : index to i64
        %39 = xegpu.create_nd_tdesc %38, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %40 = xegpu.load_nd %39[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %41 = xegpu.dpas %36, %40, %35 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %41, %23, %32 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %7 = vector.shape_cast %6#2 : vector<128xf32> to vector<128x1xf32>
      %8 = vector.broadcast %7 : vector<128x1xf32> to vector<128x64xf32>
      %9 = arith.divf %6#0, %8 : vector<128x64xf32>
      %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %10 = arith.muli %0, %c4 : index
      %11 = arith.addi %intptr_6, %10 : index
      %12 = arith.index_cast %11 : index to i64
      %13 = xegpu.create_nd_tdesc %12, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %9, %13[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}


// -----// IR Dump After LowerAffinePass (lower-affine) //----- //
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}, gpu.container_module} {
  func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}, %arg5: memref<i8>) attributes {gc.num_kernels = 1 : i32} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    gpu.launch_func  @attention_f16_kernel::@attention_f16_kernel blocks in (%c4, %c32, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4x4096x64xf16>, %arg1 : memref<4x4096x64xf16>, %arg2 : memref<4x4096x64xf16>, %arg3 : memref<4x4096x64xf32>)
    memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
    return
  }
  gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">, #xevm.target<O = 3>] {
    gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant dense<0.000000e+00> : vector<128xf32>
      %cst_0 = arith.constant dense<0xFF800000> : vector<128xf32>
      %cst_1 = arith.constant dense<0.000000e+00> : vector<128x64xf32>
      %c0 = arith.constant 0 : index
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %c262144 = arith.constant 262144 : index
      %0 = arith.muli %block_id_x, %c262144 overflow<nsw> : index
      %c8192 = arith.constant 8192 : index
      %1 = arith.muli %block_id_y, %c8192 overflow<nsw> : index
      %2 = arith.addi %0, %1 : index
      %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
      %3 = arith.muli %2, %c2 : index
      %4 = arith.addi %intptr, %3 : index
      %5 = arith.index_cast %4 : index to i64
      %6 = xegpu.create_nd_tdesc %5, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %7 = xegpu.load_nd %6[0, 0]  : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<128x64xf16>
      %8:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
        %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %c262144_11 = arith.constant 262144 : index
        %16 = arith.muli %block_id_x, %c262144_11 overflow<nsw> : index
        %c64_12 = arith.constant 64 : index
        %17 = arith.muli %arg4, %c64_12 overflow<nsw> : index
        %18 = arith.addi %16, %17 : index
        %intptr_13 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
        %19 = arith.muli %18, %c2 : index
        %20 = arith.addi %intptr_13, %19 : index
        %21 = arith.index_cast %20 : index to i64
        %22 = xegpu.create_nd_tdesc %21, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %23 = xegpu.load_nd %22[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %24 = vector.transpose %23, [1, 0] : vector<64x64xf16> to vector<64x64xf16>
        %25 = xegpu.dpas %7, %24, %cst_1 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        %26 = vector.multi_reduction <maximumf>, %25, %cst_0 [1] : vector<128x64xf32> to vector<128xf32>
        %27 = arith.maximumf %arg6, %26 : vector<128xf32>
        %28 = vector.shape_cast %27 : vector<128xf32> to vector<128x1xf32>
        %29 = vector.broadcast %28 : vector<128x1xf32> to vector<128x64xf32>
        %30 = arith.subf %25, %29 : vector<128x64xf32>
        %31 = math.exp %30 : vector<128x64xf32>
        %32 = vector.multi_reduction <add>, %31, %cst [1] : vector<128x64xf32> to vector<128xf32>
        %33 = arith.subf %arg6, %27 : vector<128xf32>
        %34 = math.exp %33 : vector<128xf32>
        %35 = arith.mulf %arg7, %34 : vector<128xf32>
        %36 = arith.addf %35, %32 : vector<128xf32>
        %37 = vector.shape_cast %34 : vector<128xf32> to vector<128x1xf32>
        %38 = vector.broadcast %37 : vector<128x1xf32> to vector<128x64xf32>
        %39 = arith.mulf %arg5, %38 : vector<128x64xf32>
        %40 = arith.truncf %31 : vector<128x64xf32> to vector<128x64xf16>
        %base_buffer_14, %offset_15, %sizes_16:3, %strides_17:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
        %intptr_18 = memref.extract_aligned_pointer_as_index %base_buffer_14 : memref<f16> -> index
        %41 = arith.addi %intptr_18, %19 : index
        %42 = arith.index_cast %41 : index to i64
        %43 = xegpu.create_nd_tdesc %42, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %44 = xegpu.load_nd %43[0, 0]  : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x64xf16>
        %45 = xegpu.dpas %40, %44, %39 : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
        scf.yield %45, %27, %36 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
      }
      %9 = vector.shape_cast %8#2 : vector<128xf32> to vector<128x1xf32>
      %10 = vector.broadcast %9 : vector<128x1xf32> to vector<128x64xf32>
      %11 = arith.divf %8#0, %10 : vector<128x64xf32>
      %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
      %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
      %12 = arith.muli %2, %c4 : index
      %13 = arith.addi %intptr_6, %12 : index
      %14 = arith.index_cast %13 : index to i64
      %15 = xegpu.create_nd_tdesc %14, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %11, %15[0, 0]  : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
}


// -----// IR Dump After GpuAsyncRegionPass (gpu-async-region) //----- //
func.func @attention_f16(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>, %arg4: memref<4x4096x64xf32> {bufferize.result}, %arg5: memref<i8>) attributes {gc.num_kernels = 1 : i32} {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %0 = gpu.wait async
  %1 = gpu.launch_func async [%0] @attention_f16_kernel::@attention_f16_kernel blocks in (%c4, %c32, %c1) threads in (%c1, %c1, %c1)  args(%arg0 : memref<4x4096x64xf16>, %arg1 : memref<4x4096x64xf16>, %arg2 : memref<4x4096x64xf16>, %arg3 : memref<4x4096x64xf32>)
  gpu.wait [%1]
  memref.copy %arg3, %arg4 : memref<4x4096x64xf32> to memref<4x4096x64xf32>
  return
}

// -----// IR Dump After XeGPUPropagateLayout (xegpu-propagate-layout) //----- //
gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">, #xevm.target<O = 3>] {
  gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} dense<0.000000e+00> : vector<128xf32>
    %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} dense<0xFF800000> : vector<128xf32>
    %cst_1 = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<0.000000e+00> : vector<128x64xf32>
    %c0 = arith.constant 0 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
    %c262144 = arith.constant 262144 : index
    %0 = arith.muli %block_id_x, %c262144 overflow<nsw> : index
    %c8192 = arith.constant 8192 : index
    %1 = arith.muli %block_id_y, %c8192 overflow<nsw> : index
    %2 = arith.addi %0, %1 : index
    %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
    %3 = arith.muli %2, %c2 : index
    %4 = arith.addi %intptr, %3 : index
    %5 = arith.index_cast %4 : index to i64
    %6 = xegpu.create_nd_tdesc %5, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %7 = xegpu.load_nd %6[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<128x64xf16>
    %8:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %c262144_11 = arith.constant 262144 : index
      %16 = arith.muli %block_id_x, %c262144_11 overflow<nsw> : index
      %c64_12 = arith.constant 64 : index
      %17 = arith.muli %arg4, %c64_12 overflow<nsw> : index
      %18 = arith.addi %16, %17 : index
      %intptr_13 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
      %19 = arith.muli %18, %c2 : index
      %20 = arith.addi %intptr_13, %19 : index
      %21 = arith.index_cast %20 : index to i64
      %22 = xegpu.create_nd_tdesc %21, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2], order = [0, 1]>>
      %23 = xegpu.load_nd %22[0, 0] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2], order = [0, 1]>}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2], order = [0, 1]>> -> vector<64x64xf16>
      %24 = vector.transpose %23, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<64x64xf16> to vector<64x64xf16>
      %25 = xegpu.dpas %7, %24, %cst_1 {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      %26 = vector.multi_reduction <maximumf>, %25, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} [1] : vector<128x64xf32> to vector<128xf32>
      %27 = arith.maximumf %arg6, %26 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %28 = vector.shape_cast %27 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
      %29 = vector.broadcast %28 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
      %30 = arith.subf %25, %29 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %31 = math.exp %30 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %32 = vector.multi_reduction <add>, %31, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} [1] : vector<128x64xf32> to vector<128xf32>
      %33 = arith.subf %arg6, %27 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %34 = math.exp %33 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %35 = arith.mulf %arg7, %34 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %36 = arith.addf %35, %32 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %37 = vector.shape_cast %34 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
      %38 = vector.broadcast %37 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
      %39 = arith.mulf %arg5, %38 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %40 = arith.truncf %31 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32> to vector<128x64xf16>
      %base_buffer_14, %offset_15, %sizes_16:3, %strides_17:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %intptr_18 = memref.extract_aligned_pointer_as_index %base_buffer_14 : memref<f16> -> index
      %41 = arith.addi %intptr_18, %19 : index
      %42 = arith.index_cast %41 : index to i64
      %43 = xegpu.create_nd_tdesc %42, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
      %44 = xegpu.load_nd %43[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<64x64xf16>
      %45 = xegpu.dpas %40, %44, %39 {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      scf.yield %45, %27, %36 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_1 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>, layout_result_2 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
    %9 = vector.shape_cast %8#2 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
    %10 = vector.broadcast %9 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
    %11 = arith.divf %8#0, %10 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
    %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
    %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
    %12 = arith.muli %2, %c4 : index
    %13 = arith.addi %intptr_6, %12 : index
    %14 = arith.index_cast %13 : index to i64
    %15 = xegpu.create_nd_tdesc %14, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %11, %15[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----// IR Dump After XeGPUPeepHoleOptimizer (xegpu-optimize-peephole) //----- //
gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">, #xevm.target<O = 3>] {
  gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %cst = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} dense<0.000000e+00> : vector<128xf32>
    %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} dense<0xFF800000> : vector<128xf32>
    %cst_1 = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<0.000000e+00> : vector<128x64xf32>
    %c0 = arith.constant 0 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
    %c262144 = arith.constant 262144 : index
    %0 = arith.muli %block_id_x, %c262144 overflow<nsw> : index
    %c8192 = arith.constant 8192 : index
    %1 = arith.muli %block_id_y, %c8192 overflow<nsw> : index
    %2 = arith.addi %0, %1 : index
    %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
    %3 = arith.muli %2, %c2 : index
    %4 = arith.addi %intptr, %3 : index
    %5 = arith.index_cast %4 : index to i64
    %6 = xegpu.create_nd_tdesc %5, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %7 = xegpu.load_nd %6[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<128x64xf16>
    %8:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_1, %arg6 = %cst_0, %arg7 = %cst) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      %base_buffer_7, %offset_8, %sizes_9:3, %strides_10:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %c262144_11 = arith.constant 262144 : index
      %16 = arith.muli %block_id_x, %c262144_11 overflow<nsw> : index
      %c64_12 = arith.constant 64 : index
      %17 = arith.muli %arg4, %c64_12 overflow<nsw> : index
      %18 = arith.addi %16, %17 : index
      %intptr_13 = memref.extract_aligned_pointer_as_index %base_buffer_7 : memref<f16> -> index
      %19 = arith.muli %18, %c2 : index
      %20 = arith.addi %intptr_13, %19 : index
      %21 = arith.index_cast %20 : index to i64
      %c64_14 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %22 = arith.shrui %c64_14, %c1 : index
      %c64_15 = arith.constant 64 : index
      %c1_16 = arith.constant 1 : index
      %23 = arith.shrui %c64_15, %c1_16 : index
      %24 = xegpu.create_nd_tdesc %21, shape : [64, %22], strides : [%23, 1] : i64 -> !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>>
      %c0_17 = arith.constant 0 : index
      %c1_18 = arith.constant 1 : index
      %25 = arith.shrui %c0_17, %c1_18 : index
      %cst_19 = arith.constant dense<0> : vector<64x32xi32>
      %c0_20 = arith.constant 0 : index
      %c0_21 = arith.constant 0 : index
      %26 = arith.addi %c0_20, %c0_21 : index
      %c0_22 = arith.constant 0 : index
      %27 = arith.addi %25, %c0_22 : index
      %28 = xegpu.load_nd %24[%26, %27] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %29 = vector.insert_strided_slice %28, %cst_19 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 0], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %c0_23 = arith.constant 0 : index
      %30 = arith.addi %c0_20, %c0_23 : index
      %c8 = arith.constant 8 : index
      %31 = arith.addi %25, %c8 : index
      %32 = xegpu.load_nd %24[%30, %31] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %33 = vector.insert_strided_slice %32, %29 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 8], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %c0_24 = arith.constant 0 : index
      %34 = arith.addi %c0_20, %c0_24 : index
      %c16 = arith.constant 16 : index
      %35 = arith.addi %25, %c16 : index
      %36 = xegpu.load_nd %24[%34, %35] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %37 = vector.insert_strided_slice %36, %33 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 16], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %c0_25 = arith.constant 0 : index
      %38 = arith.addi %c0_20, %c0_25 : index
      %c24 = arith.constant 24 : index
      %39 = arith.addi %25, %c24 : index
      %40 = xegpu.load_nd %24[%38, %39] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %41 = vector.insert_strided_slice %40, %37 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 24], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %c32 = arith.constant 32 : index
      %42 = arith.addi %c0_20, %c32 : index
      %c0_26 = arith.constant 0 : index
      %43 = arith.addi %25, %c0_26 : index
      %44 = xegpu.load_nd %24[%42, %43] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %45 = vector.insert_strided_slice %44, %41 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 0], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %c32_27 = arith.constant 32 : index
      %46 = arith.addi %c0_20, %c32_27 : index
      %c8_28 = arith.constant 8 : index
      %47 = arith.addi %25, %c8_28 : index
      %48 = xegpu.load_nd %24[%46, %47] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %49 = vector.insert_strided_slice %48, %45 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 8], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %c32_29 = arith.constant 32 : index
      %50 = arith.addi %c0_20, %c32_29 : index
      %c16_30 = arith.constant 16 : index
      %51 = arith.addi %25, %c16_30 : index
      %52 = xegpu.load_nd %24[%50, %51] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %53 = vector.insert_strided_slice %52, %49 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 16], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %c32_31 = arith.constant 32 : index
      %54 = arith.addi %c0_20, %c32_31 : index
      %c24_32 = arith.constant 24 : index
      %55 = arith.addi %25, %c24_32 : index
      %56 = xegpu.load_nd %24[%54, %55] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %57 = vector.insert_strided_slice %56, %53 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 24], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %58 = vector.bitcast %57 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2], order = [0, 1]>} : vector<64x32xi32> to vector<64x64xf16>
      %59 = vector.transpose %58, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<64x64xf16> to vector<64x64xf16>
      %60 = xegpu.dpas %7, %59, %cst_1 {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      %61 = vector.multi_reduction <maximumf>, %60, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} [1] : vector<128x64xf32> to vector<128xf32>
      %62 = arith.maximumf %arg6, %61 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %63 = vector.shape_cast %62 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
      %64 = vector.broadcast %63 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
      %65 = arith.subf %60, %64 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %66 = math.exp %65 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %67 = vector.multi_reduction <add>, %66, %cst {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} [1] : vector<128x64xf32> to vector<128xf32>
      %68 = arith.subf %arg6, %62 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %69 = math.exp %68 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %70 = arith.mulf %arg7, %69 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %71 = arith.addf %70, %67 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %72 = vector.shape_cast %69 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
      %73 = vector.broadcast %72 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
      %74 = arith.mulf %arg5, %73 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %75 = arith.truncf %66 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32> to vector<128x64xf16>
      %base_buffer_33, %offset_34, %sizes_35:3, %strides_36:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %intptr_37 = memref.extract_aligned_pointer_as_index %base_buffer_33 : memref<f16> -> index
      %76 = arith.addi %intptr_37, %19 : index
      %77 = arith.index_cast %76 : index to i64
      %78 = xegpu.create_nd_tdesc %77, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
      %79 = xegpu.load_nd %78[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<64x64xf16>
      %80 = xegpu.dpas %75, %79, %74 {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      scf.yield %80, %62, %71 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_1 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>, layout_result_2 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
    %9 = vector.shape_cast %8#2 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
    %10 = vector.broadcast %9 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
    %11 = arith.divf %8#0, %10 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
    %base_buffer_2, %offset_3, %sizes_4:3, %strides_5:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
    %intptr_6 = memref.extract_aligned_pointer_as_index %base_buffer_2 : memref<f32> -> index
    %12 = arith.muli %2, %c4 : index
    %13 = arith.addi %intptr_6, %12 : index
    %14 = arith.index_cast %13 : index to i64
    %15 = xegpu.create_nd_tdesc %14, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %11, %15[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----// IR Dump After CanonicalizerPass (canonicalize) //----- //
gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">, #xevm.target<O = 3>] {
  gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
    %c24 = arith.constant 24 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant dense<0> : vector<64x32xi32>
    %c8192 = arith.constant 8192 : index
    %c262144 = arith.constant 262144 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} dense<0xFF800000> : vector<128xf32>
    %cst_2 = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<0.000000e+00> : vector<128x64xf32>
    %c0 = arith.constant 0 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
    %0 = arith.muli %block_id_x, %c262144 overflow<nsw> : index
    %1 = arith.muli %block_id_y, %c8192 overflow<nsw> : index
    %2 = arith.addi %0, %1 : index
    %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
    %3 = arith.muli %2, %c2 : index
    %4 = arith.addi %intptr, %3 : index
    %5 = arith.index_cast %4 : index to i64
    %6 = xegpu.create_nd_tdesc %5, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %7 = xegpu.load_nd %6[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<128x64xf16>
    %8:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_2, %arg6 = %cst_1, %arg7 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      %base_buffer_8, %offset_9, %sizes_10:3, %strides_11:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %16 = arith.muli %block_id_x, %c262144 overflow<nsw> : index
      %17 = arith.muli %arg4, %c64 overflow<nsw> : index
      %18 = arith.addi %16, %17 : index
      %intptr_12 = memref.extract_aligned_pointer_as_index %base_buffer_8 : memref<f16> -> index
      %19 = arith.muli %18, %c2 : index
      %20 = arith.addi %intptr_12, %19 : index
      %21 = arith.index_cast %20 : index to i64
      %22 = xegpu.create_nd_tdesc %21, shape : [64, %c32], strides : [%c32, 1] : i64 -> !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>>
      %23 = xegpu.load_nd %22[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %24 = vector.insert_strided_slice %23, %cst {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 0], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %25 = xegpu.load_nd %22[%c0, %c8] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %26 = vector.insert_strided_slice %25, %24 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 8], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %27 = xegpu.load_nd %22[%c0, %c16] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %28 = vector.insert_strided_slice %27, %26 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 16], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %29 = xegpu.load_nd %22[%c0, %c24] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %30 = vector.insert_strided_slice %29, %28 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 24], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %31 = xegpu.load_nd %22[%c32, %c0] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %32 = vector.insert_strided_slice %31, %30 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 0], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %33 = xegpu.load_nd %22[%c32, %c8] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %34 = vector.insert_strided_slice %33, %32 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 8], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %35 = xegpu.load_nd %22[%c32, %c16] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %36 = vector.insert_strided_slice %35, %34 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 16], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %37 = xegpu.load_nd %22[%c32, %c24] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %38 = vector.insert_strided_slice %37, %36 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 24], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %39 = vector.bitcast %38 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2], order = [0, 1]>} : vector<64x32xi32> to vector<64x64xf16>
      %40 = vector.transpose %39, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<64x64xf16> to vector<64x64xf16>
      %41 = xegpu.dpas %7, %40, %cst_2 {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      %42 = vector.multi_reduction <maximumf>, %41, %cst_1 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} [1] : vector<128x64xf32> to vector<128xf32>
      %43 = arith.maximumf %arg6, %42 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %44 = vector.shape_cast %43 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
      %45 = vector.broadcast %44 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
      %46 = arith.subf %41, %45 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %47 = math.exp %46 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %48 = vector.multi_reduction <add>, %47, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} [1] : vector<128x64xf32> to vector<128xf32>
      %49 = arith.subf %arg6, %43 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %50 = math.exp %49 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %51 = arith.mulf %arg7, %50 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %52 = arith.addf %51, %48 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %53 = vector.shape_cast %50 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
      %54 = vector.broadcast %53 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
      %55 = arith.mulf %arg5, %54 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %56 = arith.truncf %47 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32> to vector<128x64xf16>
      %base_buffer_13, %offset_14, %sizes_15:3, %strides_16:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %intptr_17 = memref.extract_aligned_pointer_as_index %base_buffer_13 : memref<f16> -> index
      %57 = arith.addi %intptr_17, %19 : index
      %58 = arith.index_cast %57 : index to i64
      %59 = xegpu.create_nd_tdesc %58, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
      %60 = xegpu.load_nd %59[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<64x64xf16>
      %61 = xegpu.dpas %56, %60, %55 {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      scf.yield %61, %43, %52 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_1 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>, layout_result_2 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
    %9 = vector.shape_cast %8#2 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
    %10 = vector.broadcast %9 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
    %11 = arith.divf %8#0, %10 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
    %base_buffer_3, %offset_4, %sizes_5:3, %strides_6:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
    %intptr_7 = memref.extract_aligned_pointer_as_index %base_buffer_3 : memref<f32> -> index
    %12 = arith.muli %2, %c4 : index
    %13 = arith.addi %intptr_7, %12 : index
    %14 = arith.index_cast %13 : index to i64
    %15 = xegpu.create_nd_tdesc %14, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %11, %15[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----// IR Dump After CSEPass (cse) //----- //
gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">, #xevm.target<O = 3>] {
  gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
    %c24 = arith.constant 24 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant dense<0> : vector<64x32xi32>
    %c8192 = arith.constant 8192 : index
    %c262144 = arith.constant 262144 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} dense<0xFF800000> : vector<128xf32>
    %cst_2 = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<0.000000e+00> : vector<128x64xf32>
    %c0 = arith.constant 0 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
    %0 = arith.muli %block_id_x, %c262144 overflow<nsw> : index
    %1 = arith.muli %block_id_y, %c8192 overflow<nsw> : index
    %2 = arith.addi %0, %1 : index
    %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
    %3 = arith.muli %2, %c2 : index
    %4 = arith.addi %intptr, %3 : index
    %5 = arith.index_cast %4 : index to i64
    %6 = xegpu.create_nd_tdesc %5, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %7 = xegpu.load_nd %6[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<128x64xf16>
    %8:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_2, %arg6 = %cst_1, %arg7 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      %base_buffer_8, %offset_9, %sizes_10:3, %strides_11:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %16 = arith.muli %arg4, %c64 overflow<nsw> : index
      %17 = arith.addi %0, %16 : index
      %intptr_12 = memref.extract_aligned_pointer_as_index %base_buffer_8 : memref<f16> -> index
      %18 = arith.muli %17, %c2 : index
      %19 = arith.addi %intptr_12, %18 : index
      %20 = arith.index_cast %19 : index to i64
      %21 = xegpu.create_nd_tdesc %20, shape : [64, %c32], strides : [%c32, 1] : i64 -> !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>>
      %22 = xegpu.load_nd %21[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %23 = vector.insert_strided_slice %22, %cst {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 0], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %24 = xegpu.load_nd %21[%c0, %c8] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %25 = vector.insert_strided_slice %24, %23 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 8], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %26 = xegpu.load_nd %21[%c0, %c16] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %27 = vector.insert_strided_slice %26, %25 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 16], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %28 = xegpu.load_nd %21[%c0, %c24] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %29 = vector.insert_strided_slice %28, %27 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 24], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %30 = xegpu.load_nd %21[%c32, %c0] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %31 = vector.insert_strided_slice %30, %29 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 0], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %32 = xegpu.load_nd %21[%c32, %c8] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %33 = vector.insert_strided_slice %32, %31 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 8], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %34 = xegpu.load_nd %21[%c32, %c16] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %35 = vector.insert_strided_slice %34, %33 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 16], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %36 = xegpu.load_nd %21[%c32, %c24] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %37 = vector.insert_strided_slice %36, %35 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 24], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %38 = vector.bitcast %37 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2], order = [0, 1]>} : vector<64x32xi32> to vector<64x64xf16>
      %39 = vector.transpose %38, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<64x64xf16> to vector<64x64xf16>
      %40 = xegpu.dpas %7, %39, %cst_2 {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      %41 = vector.multi_reduction <maximumf>, %40, %cst_1 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} [1] : vector<128x64xf32> to vector<128xf32>
      %42 = arith.maximumf %arg6, %41 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %43 = vector.shape_cast %42 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
      %44 = vector.broadcast %43 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
      %45 = arith.subf %40, %44 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %46 = math.exp %45 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %47 = vector.multi_reduction <add>, %46, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} [1] : vector<128x64xf32> to vector<128xf32>
      %48 = arith.subf %arg6, %42 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %49 = math.exp %48 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %50 = arith.mulf %arg7, %49 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %51 = arith.addf %50, %47 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %52 = vector.shape_cast %49 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
      %53 = vector.broadcast %52 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
      %54 = arith.mulf %arg5, %53 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %55 = arith.truncf %46 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32> to vector<128x64xf16>
      %base_buffer_13, %offset_14, %sizes_15:3, %strides_16:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %intptr_17 = memref.extract_aligned_pointer_as_index %base_buffer_13 : memref<f16> -> index
      %56 = arith.addi %intptr_17, %18 : index
      %57 = arith.index_cast %56 : index to i64
      %58 = xegpu.create_nd_tdesc %57, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
      %59 = xegpu.load_nd %58[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<64x64xf16>
      %60 = xegpu.dpas %55, %59, %54 {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      scf.yield %60, %42, %51 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_1 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>, layout_result_2 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
    %9 = vector.shape_cast %8#2 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
    %10 = vector.broadcast %9 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
    %11 = arith.divf %8#0, %10 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
    %base_buffer_3, %offset_4, %sizes_5:3, %strides_6:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
    %intptr_7 = memref.extract_aligned_pointer_as_index %base_buffer_3 : memref<f32> -> index
    %12 = arith.muli %2, %c4 : index
    %13 = arith.addi %intptr_7, %12 : index
    %14 = arith.index_cast %13 : index to i64
    %15 = xegpu.create_nd_tdesc %14, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %11, %15[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

// -----// IR Dump After XeGPUPropagateLayout (xegpu-propagate-layout) //----- //
gpu.module @attention_f16_kernel [#xevm.target<O = 3, chip = "pvc">, #xevm.target<O = 3>] {
  gpu.func @attention_f16_kernel(%arg0: memref<4x4096x64xf16>, %arg1: memref<4x4096x64xf16>, %arg2: memref<4x4096x64xf16>, %arg3: memref<4x4096x64xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
    %c24 = arith.constant 24 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<0> : vector<64x32xi32>
    %0 = xegpu.convert_layout %cst <{input_layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, target_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : vector<64x32xi32>
    %c8192 = arith.constant 8192 : index
    %c262144 = arith.constant 262144 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c4096 = arith.constant 4096 : index
    %cst_0 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} dense<0.000000e+00> : vector<128xf32>
    %cst_1 = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} dense<0xFF800000> : vector<128xf32>
    %cst_2 = arith.constant {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} dense<0.000000e+00> : vector<128x64xf32>
    %c0 = arith.constant 0 : index
    %block_id_x = gpu.block_id  x
    %block_id_y = gpu.block_id  y
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg0 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
    %1 = arith.muli %block_id_x, %c262144 overflow<nsw> : index
    %2 = arith.muli %block_id_y, %c8192 overflow<nsw> : index
    %3 = arith.addi %1, %2 : index
    %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f16> -> index
    %4 = arith.muli %3, %c2 : index
    %5 = arith.addi %intptr, %4 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = xegpu.create_nd_tdesc %6, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    %8 = xegpu.load_nd %7[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : !xegpu.tensor_desc<128x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>> -> vector<128x64xf16>
    %9:3 = scf.for %arg4 = %c0 to %c4096 step %c64 iter_args(%arg5 = %cst_2, %arg6 = %cst_1, %arg7 = %cst_0) -> (vector<128x64xf32>, vector<128xf32>, vector<128xf32>) {
      %base_buffer_8, %offset_9, %sizes_10:3, %strides_11:3 = memref.extract_strided_metadata %arg1 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %17 = arith.muli %arg4, %c64 overflow<nsw> : index
      %18 = arith.addi %1, %17 : index
      %intptr_12 = memref.extract_aligned_pointer_as_index %base_buffer_8 : memref<f16> -> index
      %19 = arith.muli %18, %c2 : index
      %20 = arith.addi %intptr_12, %19 : index
      %21 = arith.index_cast %20 : index to i64
      %22 = xegpu.create_nd_tdesc %21, shape : [64, %c32], strides : [%c32, 1] : i64 -> !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>>
      %23 = xegpu.load_nd %22[%c0, %c0] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %24 = xegpu.convert_layout %23 <{input_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, target_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> : vector<32x8xi32>
      %25 = vector.insert_strided_slice %24, %0 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 0], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %26 = xegpu.load_nd %22[%c0, %c8] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %27 = xegpu.convert_layout %26 <{input_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, target_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> : vector<32x8xi32>
      %28 = vector.insert_strided_slice %27, %25 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 8], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %29 = xegpu.load_nd %22[%c0, %c16] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %30 = xegpu.convert_layout %29 <{input_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, target_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> : vector<32x8xi32>
      %31 = vector.insert_strided_slice %30, %28 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 16], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %32 = xegpu.load_nd %22[%c0, %c24] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %33 = xegpu.convert_layout %32 <{input_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, target_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> : vector<32x8xi32>
      %34 = vector.insert_strided_slice %33, %31 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [0, 24], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %35 = xegpu.load_nd %22[%c32, %c0] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %36 = xegpu.convert_layout %35 <{input_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, target_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> : vector<32x8xi32>
      %37 = vector.insert_strided_slice %36, %34 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 0], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %38 = xegpu.load_nd %22[%c32, %c8] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %39 = xegpu.convert_layout %38 <{input_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, target_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> : vector<32x8xi32>
      %40 = vector.insert_strided_slice %39, %37 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 8], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %41 = xegpu.load_nd %22[%c32, %c16] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %42 = xegpu.convert_layout %41 <{input_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, target_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> : vector<32x8xi32>
      %43 = vector.insert_strided_slice %42, %40 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 16], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %44 = xegpu.load_nd %22[%c32, %c24] <{layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>}> : !xegpu.tensor_desc<32x8xi32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>> -> vector<32x8xi32>
      %45 = xegpu.convert_layout %44 <{input_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, target_layout = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1]>}> : vector<32x8xi32>
      %46 = vector.insert_strided_slice %45, %43 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 1], order = [0, 1]>, offsets = [32, 24], strides = [1, 1]} : vector<32x8xi32> into vector<64x32xi32>
      %47 = vector.bitcast %46 {layout_result_0 = #xegpu.layout<lane_layout = [16, 1], lane_data = [1, 2], order = [0, 1]>} : vector<64x32xi32> to vector<64x64xf16>
      %48 = vector.transpose %47, [1, 0] {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>} : vector<64x64xf16> to vector<64x64xf16>
      %49 = xegpu.dpas %8, %48, %cst_2 {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      %50 = vector.multi_reduction <maximumf>, %49, %cst_1 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} [1] : vector<128x64xf32> to vector<128xf32>
      %51 = arith.maximumf %arg6, %50 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %52 = vector.shape_cast %51 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
      %53 = vector.broadcast %52 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
      %54 = arith.subf %49, %53 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %55 = math.exp %54 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %56 = vector.multi_reduction <add>, %55, %cst_0 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} [1] : vector<128x64xf32> to vector<128xf32>
      %57 = arith.subf %arg6, %51 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %58 = math.exp %57 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %59 = arith.mulf %arg7, %58 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %60 = arith.addf %59, %56 {layout_result_0 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>} : vector<128xf32>
      %61 = vector.shape_cast %58 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
      %62 = vector.broadcast %61 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
      %63 = arith.mulf %arg5, %62 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
      %64 = arith.truncf %55 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32> to vector<128x64xf16>
      %base_buffer_13, %offset_14, %sizes_15:3, %strides_16:3 = memref.extract_strided_metadata %arg2 : memref<4x4096x64xf16> -> memref<f16>, index, index, index, index, index, index, index
      %intptr_17 = memref.extract_aligned_pointer_as_index %base_buffer_13 : memref<f16> -> index
      %65 = arith.addi %intptr_17, %19 : index
      %66 = arith.index_cast %65 : index to i64
      %67 = xegpu.create_nd_tdesc %66, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>>
      %68 = xegpu.load_nd %67[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>}> : !xegpu.tensor_desc<64x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>> -> vector<64x64xf16>
      %69 = xegpu.dpas %64, %68, %63 {layout_a = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_b = #xegpu.layout<lane_layout = [1, 16], lane_data = [2, 1]>, layout_cd = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf16>, vector<64x64xf16>, vector<128x64xf32> -> vector<128x64xf32>
      scf.yield %69, %51, %60 : vector<128x64xf32>, vector<128xf32>, vector<128xf32>
    } {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, layout_result_1 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>, layout_result_2 = #xegpu.slice<#xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>, dims = [1]>}
    %10 = vector.shape_cast %9#2 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128xf32> to vector<128x1xf32>
    %11 = vector.broadcast %10 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x1xf32> to vector<128x64xf32>
    %12 = arith.divf %9#0, %11 {layout_result_0 = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>} : vector<128x64xf32>
    %base_buffer_3, %offset_4, %sizes_5:3, %strides_6:3 = memref.extract_strided_metadata %arg3 : memref<4x4096x64xf32> -> memref<f32>, index, index, index, index, index, index, index
    %intptr_7 = memref.extract_aligned_pointer_as_index %base_buffer_3 : memref<f32> -> index
    %13 = arith.muli %3, %c4 : index
    %14 = arith.addi %intptr_7, %13 : index
    %15 = arith.index_cast %14 : index to i64
    %16 = xegpu.create_nd_tdesc %15, shape : [128, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    xegpu.store_nd %12, %16[0, 0] <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}> : vector<128x64xf32>, !xegpu.tensor_desc<128x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
    gpu.return
  }
}

