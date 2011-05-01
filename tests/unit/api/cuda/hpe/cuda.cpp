#ifndef USE_MULTI_CONTEXT
#include "core/hpe/Process.h"
#include "gtest/gtest.h"

#include "api/cuda/hpe/Accelerator.h"

#include <cuda.h>

using __impl::core::hpe::Process;
using gmac::cuda::hpe::Accelerator;

#if 0
void CUDA(Process &proc)
{
    ASSERT_EQ(CUDA_SUCCESS, cuInit(0));
    int deviceCount = 0;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&deviceCount));
    for(int i = 0; i < deviceCount; i++) {
        CUdevice device;
        ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&device, i));
        Accelerator *acc = new Accelerator(i, device);
        proc.addAccelerator(*acc);
        // Nedded for OpenCL code compilation
        Accelerator::addAccelerator(*acc);
    }
}
#endif
#endif
