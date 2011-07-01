#include "config/order.h"

#include "core/hpe/Process.h"
#include "api/opencl/hpe/gpu/amd/Accelerator.h"
#include "api/opencl/hpe/gpu/nvidia/Accelerator.h"

enum GMAC_LOCAL OpenCLVendor {
    VENDOR_AMD,
    VENDOR_NVIDIA,
    VENDOR_INTEL,
    VENDOR_UNKNOWN
};

enum GMAC_LOCAL OpenCLPlatform {
    PLATFORM_AMD,
    PLATFORM_NVIDIA,
    PLATFORM_INTEL,
    PLATFORM_UNKNOWN
};

std::string GMAC_LOCAL OpenCLNVidiaDevicePrefix[] = {
    "ION",
    "Tesla",
    "GeForce"
};

std::string GMAC_LOCAL OpenCLAMDDevicePrefix[] = {
    "RV870",
    "Loveland"
};

static
std::string getPlatformName(cl_platform_id id)
{
    size_t len;
    cl_int err = clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, NULL, &len);
    CFATAL(err == CL_SUCCESS);
    char *name = new char[len + 1];
    err = clGetPlatformInfo(id, CL_PLATFORM_NAME, len, name, NULL);
    CFATAL(err == CL_SUCCESS);
    name[len] = '\0';
    std::string ret(name);

    delete [] name;

    return ret;
}

#if 0
static
std::string getVendorName(cl_platform_id id)
{
    size_t len;
    cl_int err = clGetPlatformInfo(id, CL_PLATFORM_VENDOR, 0, NULL, &len);
    CFATAL(err == CL_SUCCESS);
    char *vendor = new char[len + 1];
    err = clGetPlatformInfo(id, CL_PLATFORM_VENDOR, len, vendor, NULL);
    CFATAL(err == CL_SUCCESS);
    vendor[len] = '\0';
    std::string ret(vendor);

    delete [] vendor;

    return ret;
}
#endif

static OpenCLPlatform
getPlatform(cl_platform_id id)
{
    static const std::string amd("AMD Accelerated Parallel Processing");
    static const std::string nvidia("NVIDIA CUDA");
    static const std::string intel("Intel OpenCL");

    std::string platformName = getPlatformName(id);

    if (platformName.compare(amd) == 0) {
        return PLATFORM_AMD;
    } else if (platformName.compare(nvidia) == 0) {
        return PLATFORM_NVIDIA;
    } else if (platformName.compare(intel) == 0) {
        return PLATFORM_INTEL;
    } else {
        return PLATFORM_UNKNOWN;
    }
}

#if 0
static OpenCLVendor
getVendor(cl_platform_id id)
{
    static const std::string amd("Advanced Micro Devices, Inc.");
    static const std::string nvidia("NVIDIA Corporation");
    static const std::string intel("Intel Corporation");

    std::string vendorName = getVendorName(id);

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
#endif

typedef std::pair<unsigned, unsigned> OpenCLVersion;
#if defined(_MSC_VER)
#define sscanf(...) sscanf_s(__VA_ARGS__)
#endif

static OpenCLVersion
getOpenCLVersion(cl_platform_id id)
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

    return OpenCLVersion(major, minor);
}

static bool initialized = false;
void OpenCL(gmac::core::hpe::Process &proc)
{
    TRACE(GLOBAL, "Initializing OpenCL API");
    cl_uint platformSize = 0;
    cl_int ret = CL_SUCCESS;
    ret = clGetPlatformIDs(0, NULL, &platformSize);
    CFATAL(ret == CL_SUCCESS);
    cl_platform_id * platforms = new cl_platform_id[platformSize];
    ret = clGetPlatformIDs(platformSize, platforms, NULL);
    CFATAL(ret == CL_SUCCESS);
    TRACE(GLOBAL, "Found %d OpenCL platforms", platformSize);

    unsigned n = 0;
    for(unsigned i = 0; i < platformSize; i++) {
        cl_uint deviceSize = 0;
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU,
            0, NULL, &deviceSize);
        ASSERTION(ret == CL_SUCCESS);
        cl_device_id *devices = new cl_device_id[deviceSize];
        ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU,
            deviceSize, devices, NULL);
        ASSERTION(ret == CL_SUCCESS);
        TRACE(GLOBAL, "Found %d OpenCL devices in platform %d", deviceSize, i);

        OpenCLVersion clVersion = getOpenCLVersion(platforms[i]);

        cl_context ctx;
        if (deviceSize > 0) {
            cl_context_properties prop[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0 };

            ctx = clCreateContext(prop, deviceSize, devices, NULL, NULL, &ret);
            CFATAL(ret == CL_SUCCESS, "Unable to create OpenCL context %d", ret);
        }
        for(unsigned j = 0; j < deviceSize; j++) {
            gmac::opencl::hpe::Accelerator *acc = NULL;

            switch (getPlatform(platforms[i])) {
                case PLATFORM_AMD:
                    acc = new __impl::opencl::hpe::gpu::amd::Accelerator(n++, ctx, devices[j],
                                                                         clVersion.first, clVersion.second);
                    break;
                case PLATFORM_NVIDIA:
                    acc = new __impl::opencl::hpe::gpu::nvidia::Accelerator(n++, ctx, devices[j],
                                                                            clVersion.first, clVersion.second);
                    break;
                case PLATFORM_INTEL:
                case PLATFORM_UNKNOWN:
                    FATAL("Platform not supported\n");
            }
            proc.addAccelerator(*acc);
            // Nedded for OpenCL code compilation
            __impl::opencl::hpe::Accelerator::addAccelerator(*acc);
        }
        if (deviceSize > 0) {
            ret = clReleaseContext(ctx);
            CFATAL(ret == CL_SUCCESS, "Unable to release OpenCL context after accelerator initialization");
        }
        delete[] devices;
    }
    delete[] platforms;
    initialized = true;

    __impl::opencl::hpe::Accelerator::prepareEmbeddedCLCode();
}
