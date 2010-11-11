#include "init.h"

#include "trace/Function.h"
#include "core/Process.h"
#include "api/cuda/Accelerator.h"

#include "gtest/gtest.h"

using gmac::Process;
using gmac::trace::Function;
using gmac::cuda::Accelerator;

void InitProcess()
{
    Function::init();
    Process::create<Process>();
    Process &proc = Process::getInstance();

    ASSERT_EQ(CUDA_SUCCESS, cuInit(0));

    int count = 0;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&count));
    ASSERT_GT(count, 0);

    CUdevice dev;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&dev, 0));
    Accelerator *accelerator = new Accelerator(dev, 0);
    ASSERT_TRUE(accelerator != NULL);

    proc.addAccelerator(accelerator);
}


void FiniProcess()
{
    Process::destroy();
}
