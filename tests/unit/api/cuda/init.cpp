#include "unit/init.h"
#include "core/Process.h"
#include "api/cuda/hpe/Accelerator.h"
#include "api/cuda/hpe/Context.h"
#include "api/cuda/hpe/Mode.h"
#include "gtest/gtest.h"

#include <cuda.h>

using gmac::cuda::hpe::Accelerator;
using __impl::cuda::hpe::Mode;
using __impl::cuda::hpe::Context;

void InitAccelerator()
{
    if(Accelerator_ != NULL) return;
    InitTrace();
    ASSERT_EQ(CUDA_SUCCESS, cuInit(0));
    int count = 0;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&count));
    ASSERT_GT(count, 0);

    CUdevice dev;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&dev, 0));
    Accelerator_ = new Accelerator(dev, 0);
    ASSERT_TRUE(Accelerator_ != NULL);
}

void InitContext()
{
    if(Context_ != NULL) return;
    ASSERT_TRUE(Accelerator_ != NULL);
    Context_ = new Context(dynamic_cast<Accelerator &>(*Accelerator_), Mode::getCurrent());
    ASSERT_TRUE(Context_ != NULL); 
} 

