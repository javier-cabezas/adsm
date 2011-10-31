#include "util/Logger.h"
#include "util/UniquePtr.h"

#include "opencl_util.h"

#if defined(_MSC_VER)
#define sscanf(...) sscanf_s(__VA_ARGS__)
#endif

namespace __impl { namespace hal { namespace opencl { namespace util {

std::string
get_platform_string(int string, cl_platform_id id)
{
    size_t len;
    cl_int err = clGetPlatformInfo(id, string, 0, NULL, &len);
    CFATAL(err == CL_SUCCESS);
    char *name = new char[len + 1];
    err = clGetPlatformInfo(id, string, len, name, NULL);
    CFATAL(err == CL_SUCCESS);
    name[len] = '\0';
    std::string ret(name);
    delete [] name;

    return ret;
}

std::string
getDeviceString(int string, cl_device_id id)
{
    size_t len;
    cl_int err = clGetDeviceInfo(id, string, 0, NULL, &len);
    CFATAL(err == CL_SUCCESS);
    char *name = new char[len + 1];
    err = clGetDeviceInfo(id, string, len, name, NULL);
    CFATAL(err == CL_SUCCESS);
    name[len] = '\0';
    std::string ret(name);
    delete [] name;

    return ret;
}

opencl_platform
get_platform(cl_platform_id id)
{
    static const std::string amd("AMD Accelerated Parallel Processing");
    static const std::string nvidia("NVIDIA CUDA");
    static const std::string intel("Intel OpenCL");
    static const std::string apple("Apple");

    std::string platformName = get_platform_name(id);

    if (platformName.compare(amd) == 0) {
        return PLATFORM_AMD;
    } else if (platformName.compare(nvidia) == 0) {
        return PLATFORM_NVIDIA;
    } else if (platformName.compare(intel) == 0) {
        return PLATFORM_INTEL;
    } else if (platformName.compare(apple) == 0) {
        return PLATFORM_APPLE;
    } else {
        return PLATFORM_UNKNOWN;
    }
}

opencl_vendor
get_vendor(cl_platform_id id)
{
    static const std::string amd("Advanced Micro Devices, Inc.");
    static const std::string nvidia("NVIDIA Corporation");
    static const std::string intel("Intel Corporation");

    std::string vendorName = get_platform_name(id);

    if (vendorName.compare(amd) == 0) {
        return VENDOR_AMD;
    } else if (vendorName.compare(nvidia) == 0) {
        return VENDOR_NVIDIA;
    } else if (vendorName.compare(intel) == 0) {
        return VENDOR_INTEL;
    } else {
        return VENDOR_UNKNOWN;
    }
}

opencl_version
get_opencl_version(cl_platform_id id)
{
    size_t len;
    cl_int err = clGetPlatformInfo(id, CL_PLATFORM_VERSION, 0, NULL, &len);
    CFATAL(err == CL_SUCCESS);
    char *version = new char[len + 1];
    err = clGetPlatformInfo(id, CL_PLATFORM_VERSION, len, version, NULL);
    CFATAL(err == CL_SUCCESS);
    version[len] = '\0';
    unsigned major, minor;
    sscanf(version, "OpenCL %u.%u", &major, &minor);
    delete [] version;

    return opencl_version(major, minor);
}

}}}}
