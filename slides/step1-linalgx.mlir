// Step1: this snippet shows the generated Linalgx attention op in MLIR after lowering from OpenVino IR.

#mapQ = affine_map<(batch, m, k1, k2, n) -> (batch, m, k1)>
#mapK = affine_map<(batch, m, k1, k2, n) -> (batch, k2, k1)>
#mapV = affine_map<(batch, m, k1, k2, n) -> (batch, k2, n)>
#mapS = affine_map<(batch, m, k1, k2, n) -> ()>
#mapO = affine_map<(batch, m, k1, k2, n) -> (batch, m, n)>
module attributes {gc.module = {device = {arch = "pvc", max_wg_size = 1024 : i64, sg_sizes = [16, 32]}}} {
  func.func @attention_f16(
    %arg0: tensor<4x4096x64xf16>, %arg1: tensor<4x4096x64xf16>,
    %arg2: tensor<4x4096x64xf16>, %arg3: tensor<4x4096x64xf32>
) -> tensor<4x4096x64xf32> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = linalgx.attention {indexing_maps = [#mapQ, #mapK, #mapV, #mapS, #mapO]}
         ins(%arg0, %arg1, %arg2, %cst : tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, tensor<4x4096x64xf16>, f16)
         outs(%arg3 : tensor<4x4096x64xf32>) -> tensor<4x4096x64xf32>
    return %0 : tensor<4x4096x64xf32>
  }
}