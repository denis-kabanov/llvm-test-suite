// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %opencl_lib
// %level_zero_options RUN: %CPU_RUN_PLACEHOLDER %t.out RUN:
// %GPU_RUN_PLACEHOLDER %t.out RUN: %ACC_RUN_PLACEHOLDER %t.out

// REQUIRES: (opencl && opencl_icd) || (level_zero && level_zero_dev_kit) || hip

// XFAIL: hip
// Expected failure because hip does not have atomic64 check implementation

#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <level_zero/ze_api.h>

using namespace sycl;

int main() {
  queue Queue;
  device Dev = Queue.get_device();
  bool Result;
  switch (Dev.get_backend()) {
  case backend::opencl: {
    // Get size for string of extensions
    size_t ExtSize;
    clGetDeviceInfo(get_native<backend::opencl>(Dev), CL_DEVICE_EXTENSIONS, 0,
                    nullptr, &ExtSize);
    std::string ExtStr(ExtSize, '\0');

    // Collect device extensions into string ExtStr
    clGetDeviceInfo(get_native<backend::opencl>(Dev), CL_DEVICE_EXTENSIONS,
                    ExtSize, &ExtStr.front(), nullptr);

    // Check that ExtStr has two extensions related to atomic64 support
    if (ExtStr.find("cl_khr_int64_base_atomics") == std::string::npos ||
        ExtStr.find("cl_khr_int64_extended_atomics") == std::string::npos)
      Result = false;
    else
      Result = true;
    assert(Dev.has(aspect::atomic64) == Result &&
           "The Result value differs from the implemented atomic64 check on "
           "the OpenCL backend.");
    break;
  }
  case backend::ext_oneapi_level_zero: {
    ze_device_module_properties_t Properties;
    zeDeviceGetModuleProperties(get_native<backend::ext_oneapi_level_zero>(Dev),
                                &Properties);
    if (Properties.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS)
      Result = true;
    else
      Result = false;
    assert(Dev.has(aspect::atomic64) == Result &&
           "The Result value differs from the implemented atomic64 check on "
           "the L0 backend.");
  }
  case backend::ext_oneapi_hip: {
    Dev.has(aspect::atomic64);
    break;
  }
  default:
    assert(false && "Used unexpected backend.");
  }
  return 0;
}
