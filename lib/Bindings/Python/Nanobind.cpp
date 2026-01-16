#include <nanobind/nanobind.h>

#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"
#include "gc/Utils/Error.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

namespace nb = nanobind;
using namespace mlir;
using namespace mlir::gc::gpu;

struct GpuContext {
  const OclRuntime runtime;
  OclContext oclCtx;
  MLIRContext mlirCtx{gc::getDialectRegistry()};

  static GpuContext &get() {
    static GpuContext instance{gcGetOrReport(OclRuntime::get())};
    return instance;
  }

private:
  explicit GpuContext(OclRuntime rt)
      : runtime(std::move(rt)),
        oclCtx{runtime, gcGetOrReport(runtime.createQueue())} {}
};

struct Usm {
  mutable void *ptr;
  mutable size_t size;

  Usm(void *ptr = nullptr, size_t size = 0) : ptr(ptr), size(size) {}
  Usm(size_t size, bool shared = false)
      : ptr(gcGetOrReport(shared
                              ? GpuContext::get().runtime.usmAllocShared(size)
                              : GpuContext::get().runtime.usmAllocDev(size))),
        size(size) {}
  // Non-copyable
  Usm(const Usm &) = delete;
  Usm &operator=(const Usm &) = delete;
  // Movable
  Usm(Usm &&other) noexcept : ptr(other.ptr), size(other.size) {
    other.ptr = nullptr;
  }
  Usm &operator=(Usm &&other) noexcept {
    if (this != &other) {
      if (ptr) {
        gcGetOrReport(GpuContext::get().runtime.usmFree(ptr));
      }
      ptr = other.ptr;
      size = other.size;
      other.ptr = nullptr;
      other.size = 0;
    }
    return *this;
  }

  void copyFrom(const void *src, size_t size) const {
    auto &ctx = GpuContext::get();
    gcGetOrReport(ctx.runtime.usmCpy(ctx.oclCtx, src, ptr, size));
  }

  void copyTo(void *dst, size_t size) const {
    auto &ctx = GpuContext::get();
    gcGetOrReport(ctx.runtime.usmCpy(ctx.oclCtx, ptr, dst, size));
  }

  ~Usm() {
    if (ptr) {
      gcGetOrReport(GpuContext::get().runtime.usmFree(ptr));
    }
  }
};

using GpuModule = std::shared_ptr<const OclModule>;
NB_MODULE(graph_compiler, m) {
  m.doc() = "Graph Compiler";

  m.def(
      "ualloc",
      [](size_t size, bool shared = false) { return Usm(size, shared); },
      nb::arg("size"), nb::arg("shared") = false,
      "Allocate USM memory of the given size.");

  nb::class_<GpuModule>(m, "GpuModule")
      .def(nb::new_([](nb::str mlir, bool dump = false) {
             auto &ctx = GpuContext::get();
             auto mlirMod = mlir::parseSourceString<ModuleOp>(
                 std::string(mlir.c_str()), &ctx.mlirCtx);
             if (!mlirMod) {
               throw std::runtime_error("Failed to parse MLIR module");
             }

             OclModuleBuilderOpts builderOpts;
             builderOpts.dumpIr = dump;
             builderOpts.pipeline = [](OpPassManager &pm,
                                       gc::GPUPipelineOptions &opts) {
               opts.isUsmArgs = true;
               opts.callFinish = true;
               populateGPUPipeline(pm, opts);
             };
             OclModuleBuilder builder{mlirMod, builderOpts};
             auto oclMod = gcGetOrReport(builder.build(ctx.runtime));
             assert(oclMod->isStatic);
             return oclMod;
           }),
           nb::arg("mod"), nb::arg("dump") = false)

      .def("__call__", [](const GpuModule &mod, nb::args args) {
        static nb::object torchTensor = []() -> nb::object {
          try {
            return nb::module_::import_("torch").attr("Tensor");
          } catch (...) {
            return nb::none();
          }
        }();

        SmallVector<std::tuple<Usm, nb::object, void *, nb::object>> usms;
        StaticExecutor<8> exec{mod};

        for (size_t i = 0; i < args.size(); ++i) {
          nb::object arg = args[i];
          void *ptr;
          if (nb::isinstance(arg, torchTensor)) {
            // Copy tensor to CPU and then to USM
            nb::object cpu = arg.attr("cpu")().attr("contiguous")();
            size_t size = nb::cast<size_t>(cpu.attr("element_size")() *
                                           cpu.attr("nelement")());
            auto cpuPtr = reinterpret_cast<void *>(
                nb::cast<uintptr_t>(cpu.attr("data_ptr")()));
            usms.push_back({Usm(size), arg, cpuPtr, cpu});
            std::get<0>(usms.back()).copyFrom(cpuPtr, size);
            ptr = std::get<0>(usms.back()).ptr;
          } else {
            ptr = nb::cast<Usm &>(arg).ptr;
          }
          exec.arg(ptr, true);
        }

        exec(GpuContext::get().oclCtx);

        if (!usms.empty()) {
          // Copy back to tensors
          gcGetOrReport(GpuContext::get().oclCtx.finish());
          for (auto &[usm, tensor, cpuPtr, cpu] : usms) {
            usm.copyTo(cpuPtr, usm.size);
          }
          gcGetOrReport(GpuContext::get().oclCtx.finish());
          for (auto &[usm, tensor, cpuPtr, cpu] : usms) {
            tensor.attr("copy_")(cpu);
          }
        }
      });
}
