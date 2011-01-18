#include "unit/core/Mode.h"
#include "api/cuda/Mode.h"
#include "gtest/gtest.h"

using __impl::cuda::Mode;

TEST_F(ModeTest, ModeHostMemory) {
	Mode &mode = dynamic_cast<Mode &>(*Mode_);
    hostptr_t addr = NULL;
    ASSERT_EQ(gmacSuccess, mode.hostAlloc(&addr, Size_));
    ASSERT_TRUE(addr != NULL);

    ASSERT_EQ(gmacSuccess, mode.hostFree(addr));
}


