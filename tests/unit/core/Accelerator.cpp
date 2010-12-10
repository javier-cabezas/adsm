#include "Accelerator.h"
#include "core/Accelerator.h"

TEST_F(AcceleratorTest, AcceleratorMemory) {
    int *buffer = new int[Size_];
    int *canary = new int[Size_];

    memset(buffer, 0xa5, Size_ * sizeof(int));
    memset(canary, 0x5a, Size_ * sizeof(int));
    accptr_t device = NULL;
    ASSERT_TRUE(GetAccelerator().malloc(&device, Size_ * sizeof(int)) == gmacSuccess);
    ASSERT_TRUE(device != NULL);
    ASSERT_TRUE(GetAccelerator().copyToAccelerator(device, hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);
    ASSERT_TRUE(GetAccelerator().copyToHost(hostptr_t(canary), device, Size_ * sizeof(int)) == gmacSuccess);
    ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) == 0);
    ASSERT_TRUE(GetAccelerator().free(device) == gmacSuccess);
    delete[] canary;
    delete[] buffer;
}

TEST_F(AcceleratorTest, AcceleratorAligment) {
    const int max = 32 * 1024 * 1024;
    for(int n = 1; n < max; n <<= 1) {
        accptr_t device = NULL;
        ASSERT_TRUE(GetAccelerator().malloc(&device, Size_, n) == gmacSuccess);
        ASSERT_TRUE(device != NULL);
        ASSERT_EQ(0u, (unsigned long)device % n);
        ASSERT_TRUE(GetAccelerator().free(device) == gmacSuccess);
    }
}
