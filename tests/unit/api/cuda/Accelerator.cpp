#include "gtest/gtest.h"
#include "api/cuda/Accelerator.h"

#include <cuda.h>

using gmac::cuda::Accelerator;

class AcceleratorTest : public testing::Test {
protected:
    static Accelerator **Accelerator_;
    static int DevCount_;
    static const int Size_ = 4 * 1024 * 1024;

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

TEST_F(AcceleratorTest, AcceleratorMemory) {
    int *buffer = new int[Size_];
    int *canary = new int[Size_];
    for(int i = 0; i < DevCount_; i++) {
        memset(buffer, 0xa5, Size_ * sizeof(int));
        memset(canary, 0x5a, Size_ * sizeof(int));
        int *device = NULL;
        ASSERT_TRUE(Accelerator_[i]->malloc((void **)&device, Size_ * sizeof(int)) == gmacSuccess);
        ASSERT_TRUE(device != NULL);
        ASSERT_TRUE(Accelerator_[i]->copyToAccelerator(device, buffer, Size_ * sizeof(int)) == gmacSuccess);
        ASSERT_TRUE(Accelerator_[i]->copyToHost(canary, device, Size_ * sizeof(int)) == gmacSuccess);
        ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) == 0);
        ASSERT_TRUE(Accelerator_[i]->free(device) == gmacSuccess);
    }
    delete[] canary;
    delete[] buffer;
}

TEST_F(AcceleratorTest, AcceleratorAligment) {
    const int max = 32 * 1024 * 1024;
    for(int n = 1; n < max; n <<= 1) {
        for(int i = 0; i < DevCount_; i++) {
            void *device = NULL;
            ASSERT_TRUE(Accelerator_[i]->malloc((void **)&device, Size_, n) == gmacSuccess);
            ASSERT_EQ(0u, (unsigned long)device % n);
            ASSERT_TRUE(Accelerator_[i]->free(device) == gmacSuccess);
        }
    }
}

TEST_F(AcceleratorTest, AcceleratorHost) {
    for(int i = 0; i < DevCount_; i++) {
        int *host = NULL;
        ASSERT_EQ(gmacSuccess, Accelerator_[i]->hostAlloc((void **)&host, Size_));
        ASSERT_TRUE(host != NULL);
        ASSERT_TRUE(Accelerator_[i]->hostMap(host) != NULL);
        ASSERT_EQ(gmacSuccess, Accelerator_[i]->hostFree(host));
    }
}

TEST_F(AcceleratorTest, AcceleratorMemset) {
    unsigned *host = NULL;
    host = new unsigned[Size_];
    ASSERT_TRUE(host != NULL);
    for(int i = 0; i < DevCount_; i ++) {
        unsigned *device = NULL;
        memset(host, 0x5a, Size_ * sizeof(unsigned));
        ASSERT_EQ(gmacSuccess, Accelerator_[i]->malloc((void **)&device, Size_ * sizeof(unsigned)));
        ASSERT_TRUE(device != NULL);
        ASSERT_EQ(gmacSuccess, Accelerator_[i]->memset(device, 0xa5, Size_ * sizeof(unsigned)));
        ASSERT_EQ(gmacSuccess, Accelerator_[i]->copyToHost(host, device, Size_ * sizeof(unsigned)));
        for(int j = 0; j < Size_; j++) ASSERT_EQ(0xa5a5a5a5, host[j]);
        ASSERT_EQ(gmacSuccess, Accelerator_[i]->free(device));
    }

    delete[] host;
}
