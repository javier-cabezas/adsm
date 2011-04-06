#include "config/order.h"

#include "core/hpe/Process.h"
#include "api/opencl/hpe/Accelerator.h"

#include "hpe/init.h"

namespace __impl { namespace core {

static bool initialized = false;

void apiInit(void)
{
    core::hpe::Process &proc = core::Process::getInstance<gmac::core::hpe::Process>();
    if(initialized) FATAL("GMAC double initialization not allowed");

    TRACE(GLOBAL, "Initializing OpenCL API");
    cl_uint platformSize = 0;
    cl_int ret;
    ret = clGetPlatformIDs(0, NULL, &platformSize);
    ASSERTION(ret == CL_SUCCESS);
    cl_platform_id * platforms = new cl_platform_id[platformSize];
    ret = clGetPlatformIDs(platformSize, platforms, NULL);
    ASSERTION(ret == CL_SUCCESS);
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
        for(unsigned j = 0; j < deviceSize; j++) {
            gmac::opencl::hpe::Accelerator *acc = new gmac::opencl::hpe::Accelerator(n++, platforms[i], devices[j]);
            proc.addAccelerator(*acc);
            // Nedded for OpenCL code compilation
            __impl::opencl::hpe::Accelerator::addAccelerator(*acc);
        }
        delete[] devices;
    }
    delete[] platforms;
    initialized = true;

    gmacError_t compileRet = __impl::opencl::hpe::Accelerator::prepareEmbeddedCLCode();
}

}}
