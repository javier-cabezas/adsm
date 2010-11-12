#include "unit/core/Mode.h"
#include "api/cuda/Mode.h"
#include "gtest/gtest.h"

using gmac::cuda::Mode;

TEST_F(ModeTest, ModeHostMemory) {
	Mode &mode = dynamic_cast<Mode &>(*Mode_);
    void *addr = NULL;
    ASSERT_EQ(gmacSuccess, mode.hostAlloc(&addr, Size_));
    ASSERT_TRUE(addr != NULL);

    ASSERT_EQ(gmacSuccess, mode.hostFree(addr));
}

TEST_F(ModeTest, MemorySet) {
	Mode &mode = dynamic_cast<Mode &>(*Mode_);
    void *addr = NULL;
    ASSERT_EQ(gmacSuccess, mode.malloc(&addr, Size_ * sizeof(int)));
    ASSERT_TRUE(addr != NULL);
    ASSERT_EQ(gmacSuccess, mode.memset(addr, 0x5a, Size_ * sizeof(int)));

    int *dst = new int[Size_];
    ASSERT_EQ(gmacSuccess, mode.copyToHost(dst, addr, Size_ * sizeof(int)));
    for(size_t i = 0; i < Size_; i++) ASSERT_EQ(0x5a5a5a5a, dst[i]);

    ASSERT_EQ(gmacSuccess, mode.free(addr));
    delete[] dst;
}
