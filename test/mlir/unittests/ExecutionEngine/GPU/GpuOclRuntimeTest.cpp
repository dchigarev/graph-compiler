//===-- GpuOclRuntime.cpp - -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "gc/ExecutionEngine/Driver/Driver.h"
#include "gc/Utils/Error.h"

#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <memory>

#include <mlir/Dialect/GPU/Transforms/Passes.h>

#include "gc/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include <CL/cl_ext.h>

using namespace mlir;
using namespace gc::gpu;

constexpr char addStatic[] = R"mlir(
module @test {
  func.func @entry(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<32x32xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<32x32xf32>
    %2 = tensor.empty() : tensor<32x32xf32>
    %3 = linalg.add ins(%1, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<32x32xf32>, memref<32x32xf32>) -> ()
    return
  }
}
)mlir";

constexpr char addDynamic[] = R"mlir(
module @test {
  func.func @entry(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg0 restrict : memref<?x?xf32>
    %1 = bufferization.to_tensor %arg1 restrict : memref<?x?xf32>
    %d0 = memref.dim %arg0, %c0 : memref<?x?xf32>
    %d1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32>
    %3 = linalg.add ins(%1, %0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%2 : tensor<?x?xf32>) -> tensor<?x?xf32>
    bufferization.materialize_in_destination %3 in restrict writable %arg2 : (tensor<?x?xf32>, memref<?x?xf32>) -> ()
    return
  }
}
)mlir";

template <unsigned N, unsigned M = N> struct TestAdd {
  OclRuntime runtime = gcGetOrReport(OclRuntime::get());
  cl_command_queue queue = gcGetOrReport(runtime.createQueue());

  static constexpr unsigned size = N * M;
  float *buf0 = gcGetOrReport(runtime.usmNewDev<float>(size));
  float *buf1 = gcGetOrReport(runtime.usmNewDev<float>(size));
  float *buf2 = gcGetOrReport(runtime.usmNewShared<float>(size));
  MLIRContext mlirCtx{gc::initCompilerAndGetDialects()};
  float cpuBuf1[size] = {};
  float cpuBuf2[size] = {};

  explicit TestAdd() { std::fill(cpuBuf1, cpuBuf1 + size, 2.0f); }

  virtual ~TestAdd() {
    gcGetOrReport(runtime.releaseQueue(queue));
    assert(runtime.usmFree(buf0));
    assert(runtime.usmFree(buf1));
    assert(runtime.usmFree(buf2));
  }

  virtual void exec(std::shared_ptr<const OclModule> &mod, OclContext &ctx) = 0;

  void test(const char *code) {
    OclContext ctx(runtime, queue);
    assert(runtime.usmCpy(ctx, cpuBuf1, buf0, size));
    assert(runtime.usmCpy(ctx, cpuBuf1, buf1, size));

    OclModuleBuilder builder(parse(code));
    auto mod = gcGetOrReport(builder.build(runtime));

    exec(mod, ctx);

    assert(runtime.usmCpy(ctx, buf2, cpuBuf2, size));
    gcGetOrReport(ctx.finish());

    for (unsigned i = 0; i < size; i++) {
      // std::cout << buf2[i] << " ";
      assert(buf2[i] == 4.0f);
    }
    // std::cout << "\n";

    for (float i : cpuBuf2) {
      // std::cout << cpuBuf2[i] << " ";
      assert(i == 4.0f);
    }
  }

  OwningOpRef<ModuleOp> parse(const char *code) {
    std::unique_ptr<llvm::MemoryBuffer> memBuf =
        llvm::MemoryBuffer::getMemBuffer(code);
    llvm::SourceMgr srcMgr;
    srcMgr.AddNewSourceBuffer(std::move(memBuf), SMLoc());
    return parseSourceFile<ModuleOp>(srcMgr, &mlirCtx);
  }
};

TEST(GpuOclRuntime, TestAddStatic) {
  struct TestAddStatic1 : TestAdd<32> {
    void exec(std::shared_ptr<const OclModule> &mod, OclContext &ctx) override {
      assert(mod->isStatic);
      StaticExecutor<3> exec(mod);
      exec(ctx, buf0, buf1, buf2);
      // Check if the executor is allocated on the stack
      assert(exec.isSmall());
    }
  } test1;
  test1.test(addStatic);

  struct TestAddStatic2 : TestAdd<32> {
    void exec(std::shared_ptr<const OclModule> &mod, OclContext &ctx) override {
      assert(mod->isStatic);
      StaticExecutor<3> exec(mod);
      exec.arg(buf0);
      exec.arg(buf1);
      exec.arg(buf2);
      // Check if the executor is allocated on the stack
      assert(exec.isSmall());
      exec(ctx);
    }
  } test2;
  test2.test(addStatic);
}

TEST(GpuOclRuntime, TestAddDynamic) {
  GTEST_SKIP() << "Dynamic shapes are not yet supported";
  struct TestAddDynamic : TestAdd<32, 64> {
    void exec(std::shared_ptr<const OclModule> &mod, OclContext &ctx) override {
      assert(!mod->isStatic);
      int64_t shape[] = {32, 64};
      int64_t strides[] = {64, 1};
      DynamicExecutor<24> exec(mod);
      exec.arg(buf0, 2, shape, strides);
      exec.arg(buf1, 2, shape, strides);
      exec.arg(buf2, 2, shape, strides);
      exec(ctx);
      // Check if the executor is allocated on the stack
      assert(exec.isSmall());
    }
  } test;
  test.test(addDynamic);
}
