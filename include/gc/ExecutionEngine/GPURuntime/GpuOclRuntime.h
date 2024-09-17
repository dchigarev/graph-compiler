//===-- GpuOclRuntime.h - GPU OpenCL runtime --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GC_GPUOCLRUNTIME_H
#define GC_GPUOCLRUNTIME_H

namespace mlir::gc::gpu {
constexpr char GPU_OCL_MALLOC[] = "gcGpuOclMalloc";
constexpr char GPU_OCL_DEALLOC[] = "gcGpuOclDealloc";
constexpr char GPU_OCL_MEMCPY[] = "gcGpuOclMemcpy";
constexpr char GPU_OCL_KERNEL_CREATE[] = "gcGpuOclKernelCreate";
constexpr char GPU_OCL_KERNEL_DESTROY[] = "gcGpuOclKernelDestroy";
constexpr char GPU_OCL_KERNEL_LAUNCH[] = "gcGpuOclKernelLaunch";
constexpr char GPU_OCL_MOD_DESTRUCTOR[] = "gcGpuOclModuleDestructor";
} // namespace mlir::gc::gpu

#ifndef GC_GPU_OCL_CONST_ONLY
#include <cstdarg>
#include <unordered_set>
#include <vector>

#include <CL/cl.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::gc::gpu {
struct OclDevCtxPair {
  cl_device_id device;
  cl_context context;
  explicit OclDevCtxPair(cl_device_id device, cl_context context)
      : device(device), context(context) {}

  bool operator==(const OclDevCtxPair &other) const {
    return device == other.device && context == other.context;
  }
};
} // namespace mlir::gc::gpu
template <> struct std::hash<const mlir::gc::gpu::OclDevCtxPair> {
  std::size_t
  operator()(const mlir::gc::gpu::OclDevCtxPair &pair) const noexcept {
    return std::hash<cl_device_id>()(pair.device) ^
           std::hash<cl_context>()(pair.context);
  }
}; // namespace std
namespace mlir::gc::gpu {
struct OclModule;
struct OclContext;
struct OclModuleBuilder;

struct OclRuntime {
  // Returns the available Intel GPU device ids.
  [[nodiscard]] static llvm::Expected<SmallVector<cl_device_id, 2>>
  gcIntelDevices(size_t max = std::numeric_limits<size_t>::max());

  [[nodiscard]] static llvm::Expected<OclRuntime> get();

  [[nodiscard]] static llvm::Expected<OclRuntime> get(cl_device_id device);

  [[nodiscard]] static llvm::Expected<OclRuntime> get(cl_command_queue queue);

  [[nodiscard]] static llvm::Expected<OclRuntime> get(cl_device_id device,
                                                      cl_context context);

  static bool isOutOfOrder(cl_command_queue queue);

  [[nodiscard]] cl_context getContext() const;

  [[nodiscard]] cl_device_id getDevice() const;

  [[nodiscard]] llvm::Expected<cl_command_queue>
  createQueue(bool outOfOrder = false) const;

  [[nodiscard]] static llvm::Expected<bool>
  releaseQueue(cl_command_queue queue);

  [[nodiscard]] llvm::Expected<void *> usmAllocDev(size_t size) const;

  [[nodiscard]] llvm::Expected<void *> usmAllocShared(size_t size) const;

  [[nodiscard]] llvm::Expected<bool> usmFree(const void *ptr) const;

  [[nodiscard]] llvm::Expected<bool> usmCpy(OclContext &ctx, const void *src,
                                            void *dst, size_t size) const;

  template <typename T>
  [[nodiscard]] llvm::Expected<T *> usmNewDev(size_t size) const {
    auto expected = usmAllocDev(size * sizeof(T));
    if (expected) {
      return static_cast<T *>(*expected);
    }
    return expected.takeError();
  }

  template <typename T>
  [[nodiscard]] llvm::Expected<T *> usmNewShared(size_t size) const {
    auto expected = usmAllocShared(size * sizeof(T));
    if (expected) {
      return static_cast<T *>(*expected);
    }
    return expected.takeError();
  }

  template <typename T>
  [[nodiscard]] llvm::Expected<bool> usmCpy(OclContext &ctx, const T *src,
                                            T *dst, size_t size) const {
    return usmCpy(ctx, static_cast<const void *>(src), static_cast<void *>(dst),
                  size * sizeof(T));
  }

  // Use with caution! This is safe to check validity of USM, but may be false
  // positive for any other kinds.
  bool isUsm(const void *ptr) const;

  bool operator==(const OclRuntime &other) const {
    return getDevice() == other.getDevice() &&
           getContext() == other.getContext();
  }

private:
  struct Ext;
  struct Exports;
  friend OclContext;
  friend OclModuleBuilder;
  template <unsigned N> friend struct OclModuleExecutor;
  explicit OclRuntime(const Ext &ext);
  const Ext &ext;

#ifndef NDEBUG
  static void debug(const char *file, int line, const char *msg);
#endif
};

static constexpr int64_t ZERO = 0;
static constexpr auto ZERO_PTR = const_cast<int64_t *>(&ZERO);

// NOTE: The context is mutable and not thread-safe! It's expected to be used in
// a single thread only.
struct OclContext {
  const OclRuntime &runtime;
  cl_command_queue const queue;
  // Preserve the execution order. This is required in case of out-of-order
  // execution (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE). When the execution
  // is completed, the 'lastEvent' field contains the event of the last enqueued
  // command. If this field is false, 'waitList' is ignored.
  const bool preserveOrder;
  cl_uint waitListLen;
  cl_event *waitList;
  cl_event lastEvent;

  explicit OclContext(const OclRuntime &runtime, cl_command_queue queue,
                      cl_uint waitListLen = 0, cl_event *waitList = nullptr)
      : OclContext(runtime, queue, OclRuntime::isOutOfOrder(queue), waitListLen,
                   waitList) {}

  explicit OclContext(const OclRuntime &runtime, cl_command_queue queue,
                      bool preserveOrder, cl_uint waitListLen,
                      cl_event *waitList)
      : runtime(runtime), queue(queue), preserveOrder(preserveOrder),
        waitListLen(preserveOrder ? waitListLen : 0),
        waitList(preserveOrder ? waitList : nullptr), lastEvent(nullptr),
        clPtrs(nullptr) {
    assert(!OclRuntime::isOutOfOrder(queue) || preserveOrder);
    assert(preserveOrder || (waitListLen == 0 && waitList == nullptr));
  }

  OclContext(const OclContext &) = delete;
  OclContext &operator=(const OclContext &) = delete;

  void finish();

private:
  friend OclRuntime;
  friend OclRuntime::Exports;
  template <unsigned N> friend struct OclModuleExecutor;
  std::unordered_set<void *> *clPtrs;

  void setLastEvent(cl_event event) {
    lastEvent = event;
    if (event) {
      waitListLen = 1;
      waitList = &lastEvent;
    } else {
      waitListLen = 0;
      waitList = nullptr;
    }
  }
};

struct OclModule {
  const OclRuntime runtime;

  ~OclModule();
  OclModule(const OclModule &) = delete;
  OclModule &operator=(const OclModule &) = delete;
  OclModule(const OclModule &&) = delete;
  OclModule &operator=(const OclModule &&) = delete;

private:
  friend OclModuleBuilder;
  template <unsigned N> friend struct OclModuleExecutor;
  using MainFunc = void (*)(void **);
  const MainFunc main;
  const FunctionType functionType;
  std::unique_ptr<ExecutionEngine> engine;

  explicit OclModule(const OclRuntime &runtime, const MainFunc main,
                     func::FuncOp functionOp,
                     std::unique_ptr<ExecutionEngine> engine)
      : runtime(runtime), main(main),
        functionType(functionOp.getFunctionType()), engine(std::move(engine)) {}
};

struct OclModuleBuilder {
  friend OclRuntime;
  explicit OclModuleBuilder(ModuleOp module);
  explicit OclModuleBuilder(OwningOpRef<ModuleOp> &module)
      : OclModuleBuilder(module.release()) {}

  llvm::Expected<std::shared_ptr<const OclModule>>
  build(const OclRuntime &runtime);

  llvm::Expected<std::shared_ptr<const OclModule>>
  build(cl_command_queue queue);

  llvm::Expected<std::shared_ptr<const OclModule>> build(cl_device_id device,
                                                         cl_context context);

private:
  std::shared_mutex mux;
  ModuleOp mlirModule;
  func::FuncOp functionOp;
  std::unordered_map<const OclDevCtxPair, std::shared_ptr<const OclModule>>
      cache;
  llvm::Expected<std::shared_ptr<const OclModule>>

  build(const OclRuntime::Ext &ext);
};

// The main function arguments are added in the following format -
// https://mlir.llvm.org/docs/TargetLLVMIR/#c-compatible-wrapper-emission.
// NOTE: This class is mutable and not thread-safe!
// NOTE: The argument values are not copied, only the pointers are stored!
template <unsigned N = 64> struct OclModuleExecutor {
  explicit OclModuleExecutor(std::shared_ptr<const OclModule> &mod)
      : mod(mod) {}
  OclModuleExecutor(const OclModuleExecutor &) = delete;
  OclModuleExecutor &operator=(const OclModuleExecutor &) = delete;
  OclModuleExecutor(const OclModuleExecutor &&) = delete;
  OclModuleExecutor &operator=(const OclModuleExecutor &&) = delete;

  void exec(OclContext &ctx) {
#ifndef NDEBUG
    auto rt = OclRuntime::get(ctx.queue);
    assert(rt);
    assert(*rt == mod->runtime);
#endif
    auto size = args.size();
    auto ctxPtr = &ctx;
    ctx.clPtrs = &clPtrs;
    args.emplace_back(&ctxPtr);
    args.emplace_back(&ctxPtr);
    args.emplace_back(ZERO_PTR);
    mod->main(args.data());
    args.truncate(size);
  }

  void operator()(OclContext &ctx) { exec(ctx); }

  template <typename T>
  [[nodiscard]] bool operator()(OclContext &ctx, T **ptr1, ...) {
    {
      SmallVector<int64_t> values;
      auto argTypes = mod->functionType.getInputs();
      unsigned numValues = 0;

      for (unsigned i = 0, n = argTypes.size() - 1; i < n; i++) {
        if (auto type = llvm::dyn_cast<MemRefType>(argTypes[i])) {
          if (type.hasStaticShape()) {
            numValues += type.getShape().size() * 2 + 1;
            continue;
          }
        }
#ifndef NDEBUG
        OclRuntime::debug(
            __FILE__, __LINE__,
            "Only memref arguments with static shape are supported.");
#endif
        return false;
      }

      values.reserve(numValues);
      SmallVector<int64_t> strides;
      int64_t offset;
      va_list args;
      va_start(args, ptr1);

      for (unsigned i = 0, n = argTypes.size() - 1; i < n; i++) {
        auto type = llvm::dyn_cast<MemRefType>(argTypes[i]);
        strides.clear();
        if (failed(getStridesAndOffset(type, strides, offset))) {
#ifndef NDEBUG
          OclRuntime::debug(__FILE__, __LINE__,
                            "Failed to get strides and offset.");
#endif
          return false;
        }
        auto offsetPtr = values.end();
        values.emplace_back(offset);
        auto shapePtr = values.end();
        auto shape = type.getShape();
        values.append(shape.begin(), shape.end());
        auto stridesPtr = values.end();
        values.append(strides.begin(), strides.end());
        auto ptr =
            (i == 0) ? reinterpret_cast<void **>(ptr1) : va_arg(args, void **);
        addArg(*ptr, *ptr, *offsetPtr, shape.size(), shapePtr, stridesPtr);
      }

      va_end(args);
      exec(ctx);
      return true;
    }
  }

  void addArg(void *&alignedPtr, size_t rank, const int64_t *shape,
              const int64_t *strides, bool isUsm = true) {
    addArg(alignedPtr, alignedPtr, ZERO, rank, shape, strides, isUsm);
  }

  void addArg(void *&allocatedPtr, void *&alignedPtr, const int64_t &offset,
              size_t rank, const int64_t *shape, const int64_t *strides,
              bool isUsm = true) {
#ifndef NDEBUG
    assert(!isUsm || mod->runtime.isUsm(alignedPtr));
    // It's recommended to have at least 16-byte alignment
    assert(reinterpret_cast<std::uintptr_t>(alignedPtr) % 16 == 0);
    if (auto type = llvm::dyn_cast<MemRefType>(getArgType(argCounter))) {
      if (type.hasStaticShape()) {
        auto size = type.getShape();
        assert(rank == size.size());
        for (size_t i = 0; i < rank; i++) {
          assert(shape[i] == size[i]);
        }

        SmallVector<int64_t> expectedStrides;
        if (int64_t expectedOffset; !failed(
                getStridesAndOffset(type, expectedStrides, expectedOffset))) {
          assert(expectedOffset == offset);
          for (size_t i = 0; i < rank; i++) {
            assert(expectedStrides[i] == strides[i]);
          }
        }
      }
    }

    std::ostringstream oss;
    oss << "Arg" << argCounter << ": ptr=" << allocatedPtr
        << ", alignedPtr=" << alignedPtr << ", offset=" << offset
        << ", shape=[";
    for (unsigned i = 0; i < rank; i++) {
      oss << shape[i] << (i + 1 < rank ? ", " : "]");
    }
    oss << ", strides=[";
    for (unsigned i = 0; i < rank; i++) {
      oss << strides[i] << (i + 1 < rank ? ", " : "]");
    }
    OclRuntime::debug(__FILE__, __LINE__, oss.str().c_str());
#endif

    argCounter++;
    args.emplace_back(&allocatedPtr);
    args.emplace_back(&alignedPtr);
    args.emplace_back(const_cast<int64_t *>(&offset));
    for (size_t i = 0; i < rank; i++) {
      args.emplace_back(const_cast<int64_t *>(&shape[i]));
    }
    for (size_t i = 0; i < rank; i++) {
      args.emplace_back(const_cast<int64_t *>(&strides[i]));
    }
    if (!isUsm) {
      clPtrs.insert(alignedPtr);
    }
  }

  template <typename T>
  void addArg(T *&alignedPtr, size_t rank, const int64_t *shape,
              const int64_t *strides, bool isUsm = true) {
    addArg(reinterpret_cast<void *&>(alignedPtr), rank, shape, strides, isUsm);
  }

  template <typename T>
  void addArg(T *&allocatedPtr, T *&alignedPtr, const int64_t &offset,
              size_t rank, const int64_t *shape, const int64_t *strides,
              bool isUsm = true) {
    addArg(reinterpret_cast<void *&>(allocatedPtr),
           reinterpret_cast<void *&>(alignedPtr), offset, rank, shape, strides,
           isUsm);
  }

  Type getArgType(unsigned idx) const {
    assert(idx < mod->functionType.getNumInputs() - 1);
    return mod->functionType.getInput(idx);
  }

  void reset() {
    args.clear();
    clPtrs.clear();
    argCounter = 0;
  }

private:
  const std::shared_ptr<const OclModule> &mod;
  // Contains the pointers of all non-USM arguments. It's expected, that the
  // arguments are either USM or CL pointers and most probably are USM, thus,
  // in most cases, this set will be empty.
  std::unordered_set<void *> clPtrs;
  SmallVector<void *, N + 3> args;
  unsigned argCounter = 0;
};
}; // namespace mlir::gc::gpu
#else
#undef GC_GPU_OCL_CONST_ONLY
#endif
#endif
