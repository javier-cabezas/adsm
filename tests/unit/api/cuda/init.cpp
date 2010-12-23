#include "unit/init.h"
#include "api/cuda/Accelerator.h"
#include "gtest/gtest.h"

#include <cuda.h>

using gmac::cuda::Accelerator;

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
