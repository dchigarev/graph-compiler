//===-- GpuOclRuntime.cpp - GPU OpenCL Runtime ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/cl_ext.h>

// Comment out the following line to enable debug logging
// #define GC_LOG_NO_DEBUG
#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "gc/Transforms/Passes.h"
#include "gc/Utils/Error.h"
#include "gc/Utils/Log.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Error.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"

#include <llvm/ADT/SmallSet.h>

namespace mlir::gc::gpu {

#define makeClErrPref(code) "OpenCL error ", code, ": "
#define makeClErr(code, ...) gcMakeErr(makeClErrPref(code), __VA_ARGS__)
#define reportClErr(code, ...) gcReportErr(makeClErrPref(code), __VA_ARGS__)

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond))                                                               \
      return gcMakeErr(__VA_ARGS__);                                           \
  } while (0)
#define CHECKE(expected, ...)                                                  \
  do {                                                                         \
    if (!expected) {                                                           \
      gcLogE(__VA_ARGS__);                                                     \
      return expected.takeError();                                             \
    }                                                                          \
  } while (0)
#define CL_CHECK(expr, ...)                                                    \
  do {                                                                         \
    if (auto _cl_check_err = (expr); _cl_check_err != CL_SUCCESS)              \
      return makeClErr(_cl_check_err, __VA_ARGS__);                            \
  } while (0)
#define CL_CHECKR(expr, ...)                                                   \
  do {                                                                         \
    if (auto _cl_check_err = (expr); _cl_check_err != CL_SUCCESS) {            \
      reportClErr(_cl_check_err, __VA_ARGS__);                                 \
    }                                                                          \
  } while (0)

// cl_ext function pointers
struct OclRuntime::Ext : OclDevCtxPair {
  clDeviceMemAllocINTEL_fn clDeviceMemAllocINTEL;
  clSharedMemAllocINTEL_fn clSharedMemAllocINTEL;
  clMemFreeINTEL_fn clMemFreeINTEL;
  clEnqueueMemcpyINTEL_fn clEnqueueMemcpyINTEL;
  clGetMemAllocInfoINTEL_fn clGetMemAllocInfoINTEL;
  clSetKernelArgMemPointerINTEL_fn clSetKernelArgMemPointerINTEL;

  explicit Ext(cl_device_id device, cl_context context,
               clDeviceMemAllocINTEL_fn clDeviceMemAllocINTEL,
               clSharedMemAllocINTEL_fn clSharedMemAllocINTEL,
               clMemFreeINTEL_fn clMemFreeINTEL,
               clEnqueueMemcpyINTEL_fn clEnqueueMemcpyINTEL,
               clGetMemAllocInfoINTEL_fn clGetMemAllocInfoINTEL,
               clSetKernelArgMemPointerINTEL_fn clSetKernelArgMemPointerINTEL)
      : OclDevCtxPair(device, context),
        clDeviceMemAllocINTEL(clDeviceMemAllocINTEL),
        clSharedMemAllocINTEL(clSharedMemAllocINTEL),
        clMemFreeINTEL(clMemFreeINTEL),
        clEnqueueMemcpyINTEL(clEnqueueMemcpyINTEL),
        clGetMemAllocInfoINTEL(clGetMemAllocInfoINTEL),
        clSetKernelArgMemPointerINTEL(clSetKernelArgMemPointerINTEL) {}

  static llvm::Expected<const Ext *> get(cl_device_id device,
                                         cl_context context) {
    static std::shared_mutex mux;
    static std::unordered_map<const OclDevCtxPair, const Ext *> cache;

    OclDevCtxPair pair{device, context};
    {
      std::shared_lock<std::shared_mutex> lock(mux);
      if (auto it = cache.find(pair); it != cache.end()) {
        return it->second;
      }
    }

    cl_platform_id platform;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                             &platform, nullptr),
             "Failed to get the device platform.");

#define FIND_FUNC(name)                                                        \
  auto name = reinterpret_cast<name##_fn>(                                     \
      clGetExtensionFunctionAddressForPlatform(platform, #name));              \
  CHECK(name, "Failed to get the " #name " function address.")

    FIND_FUNC(clDeviceMemAllocINTEL);
    FIND_FUNC(clSharedMemAllocINTEL);
    FIND_FUNC(clMemFreeINTEL);
    FIND_FUNC(clEnqueueMemcpyINTEL);
    FIND_FUNC(clGetMemAllocInfoINTEL);
    FIND_FUNC(clSetKernelArgMemPointerINTEL);

    std::lock_guard<std::shared_mutex> lock(mux);
    if (auto it = cache.find(pair); it != cache.end()) {
      return it->second;
    }
    auto ext =
        new Ext(device, context, clDeviceMemAllocINTEL, clSharedMemAllocINTEL,
                clMemFreeINTEL, clEnqueueMemcpyINTEL, clGetMemAllocInfoINTEL,
                clSetKernelArgMemPointerINTEL);
    cache.emplace(pair, ext);
    return ext;
  }
};

struct Kernel {
  cl_program program;
  cl_kernel kernel;
  const size_t globalSize[3];
  const size_t localSize[3];
  const SmallVector<size_t> argSize;

  explicit Kernel(cl_program program, cl_kernel kernel, const size_t *gridSize,
                  const size_t *blockSize, size_t argNum, const size_t *argSize)
      : program(program),
        kernel(kernel), globalSize{gridSize[0] * blockSize[0],
                                   gridSize[1] * blockSize[1],
                                   gridSize[2] * blockSize[2]},
        localSize{blockSize[0], blockSize[1], blockSize[2]},
        argSize(argSize, argSize + argNum) {
#ifndef NDEBUG
    std::string args;
    for (size_t i = 0; i < argNum; i++) {
      args += std::to_string(argSize[i]);
      if (i < argNum - 1) {
        args += ", ";
      }
    }
    gcLogD("Kernel ", kernel, " params: globalSize=[", globalSize[0], ", ",
           globalSize[1], ", ", globalSize[2], "], localSize=[", localSize[0],
           ", ", localSize[1], ", ", localSize[2], "], argSize=[", args.c_str(),
           "]");
#endif
  }

  ~Kernel() {
    CL_CHECKR(clReleaseKernel(kernel), "Failed to release OpenCL kernel.");
    gcLogD("Released OpenCL kernel: ", kernel);
    CL_CHECKR(clReleaseProgram(program), "Failed to release OpenCL program.");
    gcLogD("Released OpenCL program: ", program);
  }
};

// Functions exported to the ExecutionEngine
struct OclRuntime::Exports {
  static llvm::orc::SymbolMap symbolMap(llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap map;
    map.reserve(6);
    map.try_emplace(interner(GPU_OCL_MALLOC),
                    llvm::orc::ExecutorAddr::fromPtr(&alloc),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_DEALLOC),
                    llvm::orc::ExecutorAddr::fromPtr(&dealloc),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_MEMCPY),
                    llvm::orc::ExecutorAddr::fromPtr(&memcpy),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_KERNEL_CREATE),
                    llvm::orc::ExecutorAddr::fromPtr(&kernelCreate),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_KERNEL_DESTROY),
                    llvm::orc::ExecutorAddr::fromPtr(&kernelDestroy),
                    llvm::JITSymbolFlags::Exported);
    map.try_emplace(interner(GPU_OCL_KERNEL_LAUNCH),
                    llvm::orc::ExecutorAddr::fromPtr(&kernelLaunch),
                    llvm::JITSymbolFlags::Exported);
    return map;
  }

private:
  static void *alloc(const OclContext *ctx, size_t size) {
    return gcGetOrReport(ctx->runtime.usmAllocDev(size));
  }

  static void dealloc(const OclContext *ctx, const void *ptr) {
    gcGetOrReport(ctx->runtime.usmFree(ptr));
  }

  static void memcpy(OclContext *ctx, const void *src, void *dst, size_t size) {
    gcGetOrReport(ctx->runtime.usmCpy(*ctx, src, dst, size));
  }

  static Kernel *kernelCreate(const OclContext *ctx, size_t spirvLen,
                              const unsigned char *spirv, const char *name,
                              const size_t *gridSize, const size_t *blockSize,
                              size_t argNum, const size_t *argSize) {
    cl_int err;
    auto program =
        clCreateProgramWithIL(ctx->runtime.ext.context, spirv, spirvLen, &err);
    CL_CHECKR(err, "Failed to create OpenCL program with IL.");

    gcLogD("Created new OpenCL program: ", program);
    clBuildProgram(program, 1, &ctx->runtime.ext.device, nullptr, nullptr,
                   nullptr);
    CL_CHECKR(err, "Failed to build the program: ", program);
    gcLogD("The program has been built: ", program);

    auto kernel = clCreateKernel(program, name, &err);
    CL_CHECKR(err, "Failed to create OpenCL kernel from program: ", program);
    gcLogD("Created new OpenCL kernel ", kernel, " from program ", program);

    cl_bool enable = CL_TRUE;
    err = clSetKernelExecInfo(kernel,
                              CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
                              sizeof(enable), &enable);
    CL_CHECKR(err, "Failed to set indirect host access.");
    err = clSetKernelExecInfo(kernel,
                              CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
                              sizeof(enable), &enable);
    CL_CHECKR(err, "Failed to set indirect device access.");
    err = clSetKernelExecInfo(kernel,
                              CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
                              sizeof(enable), &enable);
    CL_CHECKR(err, "Failed to set indirect shared access.");

    return new Kernel(program, kernel, gridSize, blockSize, argNum, argSize);
  }

  static void kernelDestroy(size_t count, Kernel **kernels) {
    gcLogD("Destroying kernels.");
    for (size_t i = 0; i < count; i++) {
      if (kernels[i]) {
        delete kernels[i];
      }
    }
  }

  static void kernelLaunch(OclContext *ctx, Kernel *kernel, ...) {
    struct ClonedKernel {
      cl_kernel kernel;

      explicit ClonedKernel(cl_kernel kernel) : kernel(kernel) {}

      ~ClonedKernel() {
        gcLogD("Releasing cloned OpenCL kernel: ", kernel);
        clReleaseKernel(kernel);
      }
    };

    gcLogD("Launching kernel: ", kernel->kernel);

    cl_int err;
    ClonedKernel cloned{clCloneKernel(kernel->kernel, &err)};
    CL_CHECKR(err, "Failed to clone OpenCL kernel: ", kernel->kernel);
    gcLogD("Cloned OpenCL kernel ", kernel->kernel, ": ", cloned.kernel);

    va_list args;
    va_start(args, kernel);
    for (size_t i = 0, n = kernel->argSize.size(); i < n; i++) {
      auto size = kernel->argSize[i];
      void *ptr = va_arg(args, void *);

      if (size) {
        gcLogD("Setting kernel ", cloned.kernel, " argument ", i, " to ",
               *static_cast<int64_t *>(ptr));
        err = clSetKernelArg(cloned.kernel, i, size, ptr);
      } else if (ctx->clPtrs->find(ptr) == ctx->clPtrs->end()) {
        gcLogD("Setting kernel ", cloned.kernel, " argument ", i,
               " to USM pointer ", ptr);
        err = ctx->runtime.ext.clSetKernelArgMemPointerINTEL(cloned.kernel, i,
                                                             ptr);
      } else {
        gcLogD("Setting kernel ", cloned.kernel, " argument ", i,
               " to CL pointer ", ptr);
        err = clSetKernelArg(cloned.kernel, i, sizeof(cl_mem), ptr);
      }

      if (err != CL_SUCCESS) {
        reportClErr("Failed to set kernel ", cloned.kernel, " argument ", i,
                    " of size ", size, " CL_Error: ", err);
      }
    }
    va_end(args);

    if (ctx->preserveOrder) {
      cl_event event = nullptr;
      err = clEnqueueNDRangeKernel(ctx->queue, cloned.kernel, 3, nullptr,
                                   kernel->globalSize, kernel->localSize,
                                   ctx->waitListLen, ctx->waitList, &event);
      ctx->setLastEvent(event);
    } else {
      err = clEnqueueNDRangeKernel(ctx->queue, cloned.kernel, 3, nullptr,
                                   kernel->globalSize, kernel->localSize, 0,
                                   nullptr, nullptr);
    }

    CL_CHECKR(err, "Failed to enqueue kernel execution: ", cloned.kernel);
    gcLogD("Enqueued kernel execution: ", cloned.kernel);
  }
};

OclRuntime::OclRuntime(const Ext &ext) : ext(ext) {}

llvm::Expected<SmallVector<cl_device_id, 2>>
OclRuntime::gcIntelDevices(size_t max) {
  SmallVector<cl_device_id, 2> intelDevices;
  if (max == 0) {
    return intelDevices;
  }

  cl_uint numPlatforms;
  CL_CHECK(clGetPlatformIDs(0, nullptr, &numPlatforms),
           "Failed to get the number of platforms.");

  if (numPlatforms == 0) {
    gcLogD("No platforms found.");
    return intelDevices;
  }

  SmallVector<cl_platform_id> platforms(numPlatforms);
  auto err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  if (err != CL_SUCCESS) {
    gcLogE("Failed to get the platform ids. Error: ", err);
    return intelDevices;
  }

  for (auto platform : platforms) {
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    if (err != CL_SUCCESS) {
      gcLogE("Failed to get the number of devices on the platform.", platform,
             " Error: ", err);
      continue;
    }
    if (numDevices == 0) {
      continue;
    }

    SmallVector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices,
                         devices.data(), nullptr);
    if (err != CL_SUCCESS) {
      gcLogE("Failed to get the device ids on the platform ", platform,
             ". Error: ", err);
      continue;
    }

    for (auto dev : devices) {
      cl_uint vendorId;
      err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR_ID, sizeof(cl_uint),
                            &vendorId, nullptr);
      if (err != CL_SUCCESS) {
        gcLogE("Failed to get info about the device ", dev, ". Error: ", err);
        continue;
      }
      if (vendorId == 0x8086) {
        intelDevices.emplace_back(dev);
#ifndef NDEBUG
        size_t nameSize;
        std::string name;
        clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &nameSize);
        name.resize(nameSize);
        clGetDeviceInfo(dev, CL_DEVICE_NAME, nameSize, &name[0], nullptr);
        gcLogD("[ INFO ] GPU device ", name.c_str(), " id: ", dev);
#endif
        if (intelDevices.size() == max) {
          return intelDevices;
        }
      }
    }
  }

  return intelDevices;
}

llvm::Expected<OclRuntime> OclRuntime::get() {
  static OclRuntime *defaultRuntimePtr = nullptr;
  if (OclRuntime *rt = defaultRuntimePtr) {
    return *rt;
  }

  auto devices = gcIntelDevices(1);
  CHECKE(devices, "Failed to get Intel GPU devices.");
  if (devices->empty()) {
    return gcMakeErr("No Intel GPU devices found.");
  }

  auto rt = get(devices.get()[0]);
  CHECKE(rt, "Failed to create OclRuntime.");

  static OclRuntime defaultRuntime = rt.get();
  defaultRuntimePtr = &defaultRuntime;
  return defaultRuntime;
}

llvm::Expected<OclRuntime> OclRuntime::get(cl_device_id device) {
  static std::shared_mutex mux;
  static std::unordered_map<cl_device_id, cl_context> cache;
  cl_context context = nullptr;

  {
    std::shared_lock<std::shared_mutex> lock(mux);
    if (auto it = cache.find(device); it != cache.end()) {
      context = it->second;
    }
  }

  if (context) {
    return get(device, context);
  }

  cl_int err;
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  CL_CHECK(err, "Failed to create OpenCL context.");
  gcLogD("Created new OpenCL context: ", context);

  {
    std::lock_guard<std::shared_mutex> lock(mux);
    if (auto it = cache.find(device); it != cache.end()) {
      if (clReleaseContext(context) != CL_SUCCESS) {
        gcLogE("Failed to release OpenCL context: ", context);
      } else {
        gcLogD("Released OpenCL context: ", context);
      }
      context = it->second;
    } else {
      cache.emplace(device, context);
    }
  }

  return get(device, context);
}

llvm::Expected<OclRuntime> OclRuntime::get(cl_command_queue queue) {
  cl_device_id device;
  cl_context context;
  CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id),
                                 &device, nullptr),
           "Failed to get CL_QUEUE_DEVICE.");
  CL_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context),
                                 &context, nullptr),
           "Failed to get CL_QUEUE_CONTEXT.");
  assert(device);
  assert(context);
  return get(device, context);
}

llvm::Expected<OclRuntime> OclRuntime::get(cl_device_id device,
                                           cl_context context) {
  auto ext = Ext::get(device, context);
  CHECKE(ext, "Failed to create OclRuntime::Ext.");
  return OclRuntime{*ext.get()};
}

bool OclRuntime::isOutOfOrder(cl_command_queue queue) {
  cl_command_queue_properties properties;
  cl_int err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES,
                                     sizeof(cl_command_queue_properties),
                                     &properties, nullptr);
  if (err != CL_SUCCESS) {
    gcLogE("clGetCommandQueueInfo() failed with error code ", err);
    // Enforcing out-of-order execution mode
    return true;
  }
  return properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
}

cl_context OclRuntime::getContext() const { return ext.context; }

cl_device_id OclRuntime::getDevice() const { return ext.device; }

llvm::Expected<cl_command_queue>
OclRuntime::createQueue(bool outOfOrder) const {
  cl_int err;
  cl_command_queue queue;
#ifdef CL_VERSION_2_0
  cl_queue_properties properties[] = {
      CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
      static_cast<cl_queue_properties>(outOfOrder ? 1 : 0)};
  queue = clCreateCommandQueueWithProperties(ext.context, ext.device,
                                             properties, &err);
#else
  const cl_command_queue_properties properties =
      outOfOrder ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0;
  queue = clCreateCommandQueue(context, device, properties, &err);
#endif
  CL_CHECK(err, "Failed to create ", outOfOrder ? "out-of-order " : "",
           "OpenCL command queue.");
  gcLogD("Created new ", outOfOrder ? "out-of-order " : "",
         "OpenCL command queue: ", queue);
  return queue;
}

llvm::Expected<bool> OclRuntime::releaseQueue(cl_command_queue queue) {
  CL_CHECK(clReleaseCommandQueue(queue),
           "Failed to release OpenCL command queue: ", queue);
  gcLogD("Released OpenCL command queue: ", queue);
  return true;
}

llvm::Expected<void *> OclRuntime::usmAllocDev(size_t size) const {
  cl_int err;
  void *ptr = ext.clDeviceMemAllocINTEL(ext.context, ext.device, nullptr, size,
                                        0, &err);
  CL_CHECK(err, "Failed to allocate ", size, " bytes of device USM memory.");
  gcLogD("Allocated ", size, " bytes of device USM memory: ", ptr);
  return ptr;
}

llvm::Expected<void *> OclRuntime::usmAllocShared(size_t size) const {
  cl_int err;
  void *ptr = ext.clSharedMemAllocINTEL(ext.context, ext.device, nullptr, size,
                                        0, &err);
  CL_CHECK(err, "Failed to allocate ", size, " bytes of shared USM memory.");
  gcLogD("Allocated ", size, " bytes of shared USM memory: ", ptr);
  return ptr;
}

llvm::Expected<bool> OclRuntime::usmFree(const void *ptr) const {
  CL_CHECK(ext.clMemFreeINTEL(ext.context, const_cast<void *>(ptr)),
           "Failed to free USM memory: ", ptr);
  gcLogD("Deallocated USM memory: ", ptr);
  return true;
}

llvm::Expected<bool> OclRuntime::usmCpy(OclContext &ctx, const void *src,
                                        void *dst, size_t size) const {
  cl_int err;
  if (ctx.preserveOrder) {
    cl_event event;
    err = ext.clEnqueueMemcpyINTEL(ctx.queue, false, dst, src, size,
                                   ctx.waitListLen, ctx.waitList, &event);
    ctx.setLastEvent(event);
  } else {
    err = ext.clEnqueueMemcpyINTEL(ctx.queue, false, dst, src, size, 0, nullptr,
                                   nullptr);
  }
  CL_CHECK(err, "Failed to copy ", size, " bytes from ", src, " to ", dst);
  gcLogD("Enqueued USM memory copy of ", size, " bytes from ", src, " to ",
         dst);
  return true;
}

bool OclRuntime::isUsm(const void *ptr) const {
  cl_mem_info_intel allocType;
  auto err = ext.clGetMemAllocInfoINTEL(
      ext.context, ptr, CL_MEM_ALLOC_TYPE_INTEL, sizeof(cl_mem_info_intel),
      &allocType, nullptr);
  return err == CL_SUCCESS && allocType != CL_MEM_TYPE_UNKNOWN_INTEL;
}

#ifndef NDEBUG
void OclRuntime::debug(const char *file, int line, const char *msg) {
#ifndef GC_LOG_NO_DEBUG
  mlir::gc::log::insetLog(file, line, std::cout, "DEBUG", msg);
#endif
}
#endif

llvm::Expected<bool> OclContext::finish() {
  gcLogD("Waiting for the enqueued OpenCL commands to finish: ", queue);
  CL_CHECK(clFinish(queue),
           "Failed to finish the OpenCL command queue: ", queue);
  if (preserveOrder) {
    waitListLen = 0;
    waitList = nullptr;
    lastEvent = nullptr;
  }
  return true;
}

OclModule::~OclModule() {
  assert(engine);
  auto fn = engine->lookup(GPU_OCL_MOD_DESTRUCTOR);
  if (fn) {
    reinterpret_cast<void (*)()>(fn.get())();
  } else {
    gcLogE("Module function ", GPU_OCL_MOD_DESTRUCTOR, " not found!");
  }
}

constexpr char STATIC_MAIN[] = "gcGpuOclStaticMain";

// If all arguments of 'origFunc' are memrefs with static shape, create a new
// function called STATIC_MAIN, that accepts 2 arguments: a pointer to
// OclContext and a pointer to an array, containing pointers to aligned memory
// buffers. The function will call the original function with the context,
// buffers and the offset/shape/strides, statically created from the
// memref descriptor.
bool createStaticMain(ModuleOp &module, func::FuncOp &origFunc) {
  auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(origFunc.getName());
  if (!mainFunc) {
    gcLogE("Failed to create static main, because the function '",
           origFunc.getName().begin(), "' not found!");
    return false;
  }
  mainFunc.setAlwaysInline(true);

  auto origFuncType = origFunc.getFunctionType();
  // Assuming the original function is void
  assert(origFuncType.getResults().empty());
  auto origTypes = origFuncType.getInputs();
  auto nargs = origTypes.size() - 1; // The last one is OclContext
  SmallVector<int64_t, 64> constArgs;
  unsigned argsCounter = 0;

  for (unsigned i = 0; i < nargs; ++i) {
    if (auto type = mlir::dyn_cast<MemRefType>(origTypes[i])) {
      if (!type.hasStaticShape()) {
        gcLogD("The argument ", i, " of the function ",
               origFunc.getName().begin(), " has a dynamic shape.");
        return false;
      }

      auto shape = type.getShape();
      auto offsetPtr = constArgs.end();
      constArgs.emplace_back(0);
      constArgs.append(shape.begin(), shape.end());
      if (failed(getStridesAndOffset(type, constArgs, *offsetPtr))) {
        gcLogD("Failed to get strides and offset of arg", i,
               " of the function ", origFunc.getName().begin());
        return false;
      }
      argsCounter += shape.size() * 2 + 3;
    }
  }

  auto loc = origFunc.getLoc();
  auto ctx = module.getContext();
  ctx->getOrLoadDialect<LLVM::LLVMDialect>();
  OpBuilder builder(ctx);
  auto i64Type = builder.getI64Type();
  auto ptrType = LLVM::LLVMPointerType::get(ctx);
  auto newFuncType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                 {ptrType, ptrType});
  auto newFunc = OpBuilder::atBlockEnd(module.getBody())
                     .create<LLVM::LLVMFuncOp>(loc, STATIC_MAIN, newFuncType);
  auto &entryBlock = *newFunc.addEntryBlock(builder);
  builder.setInsertionPointToStart(&entryBlock);
  Value arrayPtr = entryBlock.getArgument(1);

  std::unordered_map<int64_t, Value> constMap;
  auto createConst = [&](int64_t i) {
    if (auto v = constMap.find(i); v != constMap.end()) {
      return v->second;
    }
    return constMap
        .emplace(i, builder.create<LLVM::ConstantOp>(
                        loc, i64Type, builder.getIntegerAttr(i64Type, i)))
        .first->second;
  };
  Value zero = createConst(0);
  Value one = createConst(1);

  SmallVector<Value, 64> args;
  args.reserve(argsCounter);

  for (unsigned i = 0, j = 0; i < nargs; i++) {
    if (i != 0) {
      arrayPtr =
          builder.create<LLVM::GEPOp>(loc, ptrType, ptrType, arrayPtr, one);
    }

    auto ptr = builder.create<LLVM::LoadOp>(loc, ptrType, arrayPtr);
    args.emplace_back(ptr);
    args.emplace_back(ptr);
    args.emplace_back(createConst(constArgs[j++]));

    for (unsigned
             k = 0,
             m = 2 * mlir::cast<MemRefType>(origTypes[i]).getShape().size();
         k < m; k++) {
      args.emplace_back(createConst(constArgs[j++]));
    }
  }

  auto oclCtxArg = entryBlock.getArgument(0);
  args.emplace_back(oclCtxArg);
  args.emplace_back(oclCtxArg);
  args.emplace_back(zero);

  auto call = builder.create<LLVM::CallOp>(loc, mainFunc, args);
  builder.create<LLVM::ReturnOp>(loc, call.getResults());
  return true;
}

OclModuleBuilder::OclModuleBuilder(const ModuleOp module) : mlirModule(module) {
  mlirModule->walk([&](func::FuncOp func) {
    if (!func.isPublic()) {
      return;
    }

    // Add a new argument for OclContext.
    auto funcType = func.getFunctionType();
    auto ctx = mlirModule->getContext();
    auto argTypes = llvm::to_vector(funcType.getInputs());
    auto resultTypes = llvm::to_vector<1>(funcType.getResults());

    auto newArgType = MemRefType::get({}, IntegerType::get(ctx, 8));
    argTypes.push_back(newArgType);
    auto newFuncType = FunctionType::get(ctx, argTypes, resultTypes);
    func.setType(newFuncType);

    // Update the entry block to match the new function signature
    auto &entryBlock = func.getBody().front();
    OpBuilder builder(func.getBody());
    entryBlock.addArgument(newArgType, func.getLoc());
    functionOp = func;
  });
  assert(functionOp);
}

llvm::Expected<std::shared_ptr<const OclModule>>
OclModuleBuilder::build(const OclRuntime &runtime) {
  {
    std::shared_lock<std::shared_mutex> lock(mux);
    if (auto it = cache.find(runtime.ext); it != cache.end()) {
      return it->second;
    }
  }
  return build(runtime.ext);
}

llvm::Expected<std::shared_ptr<const OclModule>>
OclModuleBuilder::build(cl_command_queue queue) {
  auto rt = OclRuntime::get(queue);
  CHECKE(rt, "Failed to create OclRuntime.");
  return build(rt.get());
}

llvm::Expected<std::shared_ptr<const OclModule>>
OclModuleBuilder::build(cl_device_id device, cl_context context) {
  {
    OclDevCtxPair pair{device, context};
    std::shared_lock<std::shared_mutex> lock(mux);
    if (auto it = cache.find(pair); it != cache.end()) {
      return it->second;
    }
  }

  auto ext = OclRuntime::Ext::get(device, context);
  CHECKE(ext, "Failed to create OclRuntime::Ext.");
  return build(*ext.get());
}

llvm::Expected<std::shared_ptr<const OclModule>>
OclModuleBuilder::build(const OclRuntime::Ext &ext) {
  auto mod = mlirModule.clone();
  PassManager pm{mod.getContext()};
  populateGPUPipeline(pm);
  CHECK(!pm.run(mod).failed(), "GPU pipeline failed!");

  auto isStatic = createStaticMain(mod, functionOp);
  mod.dump();

  ExecutionEngineOptions opts;
  opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
#ifndef NDEBUG
  opts.enableGDBNotificationListener = false;
  opts.enablePerfNotificationListener = false;
#endif

  auto eng = ExecutionEngine::create(mod, opts);
  CHECKE(eng, "Failed to create ExecutionEngine!");
  eng->get()->registerSymbols(OclRuntime::Exports::symbolMap);

  OclModule::MainFunc main = {nullptr};

  if (isStatic) {
    auto expect = eng.get()->lookup(STATIC_MAIN);
    CHECKE(expect, "Module function '", STATIC_MAIN, "' not found!");
    main.staticMain = reinterpret_cast<OclModule::StaticMainFunc>(*expect);
  } else {
    auto funcName = functionOp.getName();
    auto expect = eng.get()->lookupPacked(funcName);
    CHECKE(expect, "Module function '", funcName.begin(), "' not found!");
    main.wrappedMain = *expect;
  }

  std::lock_guard<std::shared_mutex> lock(mux);
  if (auto it = cache.find(ext); it != cache.end()) {
    return it->second;
  }
  std::shared_ptr<const OclModule> ptr(
      new OclModule(OclRuntime(ext), isStatic, main,
                    functionOp.getFunctionType(), std::move(eng.get())));
  return cache.emplace(OclDevCtxPair(ext.device, ext.context), ptr)
      .first->second;
}
}; // namespace mlir::gc::gpu