#include "unit/init.h"
#include "core/Process.h"
#include "api/opencl/Accelerator.h"
#include "api/opencl/Context.h"
#include "api/opencl/Mode.h"
#include "gtest/gtest.h"

#include <CL/cl.h>


void InitAccelerator()
{
    if(Accelerator_ != NULL) return;
    InitTrace();
    // TODO: Create OpenCL accelerator -- pending

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
    __impl::opencl::Mode *mode_ =
        dynamic_cast<__impl::opencl::Mode *>(__impl::core::Process::getInstance().createMode(0));
    ASSERT_TRUE(mode_ != NULL);
    mode_->initThread();
    gmac::opencl::Accelerator *acc = dynamic_cast<gmac::opencl::Accelerator *>(Accelerator_);
    ASSERT_TRUE(acc != NULL);
    Context_ = new __impl::opencl::Context(*acc, *mode_);
    ASSERT_TRUE(Context_ != NULL);
}
