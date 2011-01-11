#include "config/order.h"


#include "gmac/init.h"
#include "core/Process.h"

#include "CL/cl.h"

namespace __impl { namespace core {

static bool initialized = false;

typedef struct _device_list {
    cl_platform_id platform;
    cl_device_id *devs;
    cl_uint size;
} _device_list_t;

void apiInit(void)
{
    core::Process &proc = core::Process::getInstance();
    if(initialized)
        FATAL("GMAC double initialization not allowed");

    TRACE(GLOBAL, "Initializing OpenCL API");

    int platformVectorSize = 32;
    cl_uint platformSize = 0;
    cl_platform_id * platforms = new cl_platform_id[platformVectorSize];
    ASSERTION(clGetPlatformIDs(platformVectorSize, platforms,
        &platformSize) == CL_SUCCESS);
    TRACE(GLOBAL, "Found %d OpenCL platforms", platformSize);

    _device_list_t *devices = new _device_list_t[platformSize];
    int deviceVectorSize = 32;
    for(unsigned i = 0; i < platformSize; i++) {
        devices[i].platform = platforms[i];
        devices[i].devs = new cl_device_id[deviceVectorSize];
        ASSERTION(clGetDeviceIDs(devices[i].platform, CL_DEVICE_TYPE_ALL,
            deviceVectorSize, devices[i].devs, &devices[i].size) == CL_SUCCESS);
        TRACE(GLOBAL, "Found %d OpenCL devices in platform %d",
            devices[i].size, i);
    }
    delete[] platforms;


}

} }
