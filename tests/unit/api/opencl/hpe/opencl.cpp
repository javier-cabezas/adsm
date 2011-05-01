#ifndef USE_MULTI_CONTEXT
#include "core/hpe/Process.h"
#include "gtest/gtest.h"

#include "api/opencl/hpe/Accelerator.h"

#include <CL/cl.h>

using __impl::opencl::hpe::Accelerator;
using __impl::opencl::hpe::Mode;
using __impl::opencl::hpe::Context;

void OpenCL(Process &proc)
{
    cl_uint platformSize = 0;
    cl_int ret;
    ASSERT_EQ(CL_SUCCESS, clGetPlatformIDs(0, NULL, &platformSize));
    cl_platform_id * platforms = new cl_platform_id[platformSize];
    ASSERT_EQ(CL_SUCCESS, clGetPlatformIDs(platformSize, platforms, NULL));

    unsigned n = 0;
    for(unsigned i = 0; i < platformSize; i++) {
        cl_uint deviceSize = 0;
        ASSERT_EQ(CL_SUCCESS, clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceSize));
        cl_device_id *devices = new cl_device_id[deviceSize];  
        ASSERT_EQ(CL_SUCCESS, clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceSize, devices, NULL));
        for(unsigned j = 0; j < deviceSize; j++) {
            Accelerator *acc = new Accelerator(n++, platforms[i], devices[j]);
            proc.addAccelerator(*acc);
            // Nedded for OpenCL code compilation
            Accelerator::addAccelerator(*acc);
        }
        delete[] devices;
    }
    delete[] platforms;
}
#endif
