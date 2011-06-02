#include "config/order.h"

#include "core/hpe/Process.h"
#include "api/opencl/hpe/gpu/amd/Accelerator.h"
#include "api/opencl/hpe/gpu/nvidia/Accelerator.h"

enum GMAC_LOCAL OpenCLVendor {
    AMD,
    NVIDIA,
    INTEL,
    UNKNOWN
};

static std::string getVendorName(cl_platform_id id)
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

OpenCLVendor getVendor(cl_platform_id id)
{
    static const std::string amd("Advanced Micro Devices, Inc.");
    static const std::string nvidia("NVIDIA Corporation");
    static const std::string intel("Intel Corporation");

    std::string vendorName = getVendorName(id);

    if (vendorName.compare(amd) == 0) {
        return AMD;
    } else if (vendorName.compare(nvidia) == 0) {
        return NVIDIA;
    } else if (vendorName.compare(intel) == 0) {
        return INTEL;
    } else {
        return UNKNOWN;
    }
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

        cl_context ctx;
        if (deviceSize > 0) {
            cl_context_properties prop[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[i], 0 };

            ctx = clCreateContext(prop, deviceSize, devices, NULL, NULL, &ret);
            CFATAL(ret == CL_SUCCESS, "Unable to create OpenCL context %d", ret);
        }
        for(unsigned j = 0; j < deviceSize; j++) {
            gmac::opencl::hpe::Accelerator *acc = NULL;

            switch (getVendor(platforms[i])) {
                case AMD:
                    acc = new __impl::opencl::hpe::gpu::amd::Accelerator(n++, ctx, devices[j]);
                    break;
                case NVIDIA:
                    acc = new __impl::opencl::hpe::gpu::nvidia::Accelerator(n++, ctx, devices[j]);
                    break;
                case INTEL:
                case UNKNOWN:
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
