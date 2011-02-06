#include "unit/init.h"
#include "core/Process.h"
#include "api/opencl/Accelerator.h"
#include "api/opencl/Context.h"
#include "api/opencl/Mode.h"
#include "gtest/gtest.h"

#include <CL/cl.h>

using __impl::opencl::Accelerator;
using __impl::opencl::Mode;
using __impl::opencl::Context;

void InitAccelerator()
{
    if(Accelerator_ != NULL) return;
    InitTrace();

    cl_platform_id platform;
    ASSERT_EQ(CL_SUCCESS, clGetPlatformIDs(1, &platform, NULL));

    cl_device_id device;
    ASSERT_EQ(CL_SUCCESS, clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    Accelerator_ = new gmac::opencl::Accelerator(0, platform, device);
    ASSERT_TRUE(Accelerator_ != NULL);
    gmac::opencl::Accelerator::addAccelerator(dynamic_cast<gmac::opencl::Accelerator &>(*Accelerator_));
}


void InitContext()
{
    if(Context_ != NULL) return;
    ASSERT_TRUE(Accelerator_ != NULL);
    Context_ = new Context(dynamic_cast<Accelerator &>(*Accelerator_), Mode::getCurrent());
    ASSERT_TRUE(Context_ != NULL);
}
