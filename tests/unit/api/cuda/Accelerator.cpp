#include "gtest/gtest.h"
#include "api/cuda/Accelerator.h"

#include <cuda.h>

using gmac::cuda::Accelerator;

class AcceleratorTest : public testing::Test {
protected:
    static Accelerator **Accelerator_;
    static int DevCount_;

    static void TearDownTestCase() {
        for(int i = 0; i < DevCount_; i++) delete Accelerator_[i];
        delete[] Accelerator_;
    }
};

Accelerator **AcceleratorTest::Accelerator_ = NULL;
int AcceleratorTest::DevCount_ = 0;

TEST_F(AcceleratorTest, AcceleratorCreation) {
    ASSERT_TRUE(cuInit(0) == CUDA_SUCCESS);
    ASSERT_TRUE(cuDeviceGetCount(&DevCount_) == CUDA_SUCCESS);
    ASSERT_TRUE(DevCount_ > 0);
    Accelerator_ = new Accelerator*[DevCount_];
    ASSERT_TRUE(Accelerator_ != NULL);
    for(int i = 0; i < DevCount_; i++) {
        Accelerator_[i] = NULL;
        CUdevice dev;
        ASSERT_TRUE(cuDeviceGet(&dev, i) == CUDA_SUCCESS);
        Accelerator_[i] = new Accelerator(dev, i);
        ASSERT_TRUE(Accelerator_[i] != NULL);
    }
}

