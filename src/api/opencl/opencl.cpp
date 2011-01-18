#include "config/order.h"

#include "gmac/init.h"
#include "core/Process.h"

#include "Accelerator.h"

namespace __impl { namespace core {

static bool initialized = false;

void apiInit(void)
{
    core::Process &proc = core::Process::getInstance();
    if(initialized) FATAL("GMAC double initialization not allowed");

    TRACE(GLOBAL, "Initializing OpenCL API");
    int platformVectorSize = 32;
    cl_uint platformSize = 0;
    cl_platform_id * platforms = new cl_platform_id[platformVectorSize];
    ASSERTION(clGetPlatformIDs(platformVectorSize, platforms,
        &platformSize) == CL_SUCCESS);
    TRACE(GLOBAL, "Found %d OpenCL platforms", platformSize);

    int deviceVectorSize = 32;
    unsigned n = 0;
    cl_device_id *devices = new cl_device_id[deviceVectorSize];  
    cl_uint size = 0;
    for(unsigned i = 0; i < platformSize; i++) {
        ASSERTION(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
            deviceVectorSize, devices, &size) == CL_SUCCESS);
        TRACE(GLOBAL, "Found %d OpenCL devices in platform %d", size, i);
        for(unsigned j = 0; j < size; j++) {
            __impl::opencl::Accelerator *acc = new __impl::opencl::Accelerator(n++, platforms[i], devices[j]);
            proc.addAccelerator(*acc);
            // Nedded for OpenCL code compilation
            __impl::opencl::Accelerator::addAccelerator(*acc);
        }
    }
    delete[] devices;
    delete[] platforms;
    initialized = true;
}

} }
