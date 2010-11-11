#include "init.h"

#include "core/Process.h"
#include "api/cuda/Mode.h"
#include "trace/Function.h"

#include "gtest/gtest.h"

using gmac::Process;
using gmac::cuda::Mode;

class ModeTest : public testing::Test {
public:
    static Mode *Mode_;

    const static size_t Size_ = 4 * 1024 * 1024;

    static void SetUpTestCase() {
        InitProcess();
    }

    static void TearDownTestCase() {
        FiniProcess();
    }
};

Mode *ModeTest::Mode_ = NULL;

TEST_F(ModeTest, ModeCreation) {
    Mode_ = dynamic_cast<Mode *>(Process::getInstance().createMode(0));
    ASSERT_TRUE(Mode_ != NULL);
    Mode_->initThread();
}

TEST_F(ModeTest, ModeCurrent) {
    Mode_->attach();
    Mode &current = Mode::current();
    ASSERT_TRUE(&current == Mode_);
}

TEST_F(ModeTest, ModeMemory) {
    void *addr = NULL;
    ASSERT_EQ(gmacSuccess, Mode_->malloc(&addr, Size_));
    ASSERT_TRUE(addr != NULL);

    ASSERT_EQ(gmacSuccess, Mode_->free(addr));
}

TEST_F(ModeTest, ModeHostMemory) {
    void *addr = NULL;
    ASSERT_EQ(gmacSuccess, Mode_->hostAlloc(&addr, Size_));
    ASSERT_TRUE(addr != NULL);

    ASSERT_EQ(gmacSuccess, Mode_->hostFree(addr));
}

TEST_F(ModeTest, MemoryCopy) {
    int *src = new int[Size_];
    ASSERT_TRUE(src != NULL);
    int *dst = new int[Size_];
    ASSERT_TRUE(dst != NULL);
    memset(src, 0x5a, Size_ * sizeof(int));
    void *addr = NULL;
    ASSERT_EQ(gmacSuccess, Mode_->malloc(&addr, Size_ * sizeof(int)));
    ASSERT_TRUE(addr != NULL);
    ASSERT_EQ(gmacSuccess, Mode_->copyToAccelerator(addr, src, Size_ * sizeof(int)));
    ASSERT_EQ(gmacSuccess, Mode_->copyToHost(dst, addr, Size_ * sizeof(int)));
    for(size_t i = 0; i < Size_; i++) ASSERT_EQ(0x5a5a5a5a, dst[i]);
    ASSERT_EQ(gmacSuccess, Mode_->free(addr));
    delete[] dst;
    delete[] src;
}

TEST_F(ModeTest, MemorySet) {
    void *addr = NULL;
    ASSERT_EQ(gmacSuccess, Mode_->malloc(&addr, Size_ * sizeof(int)));
    ASSERT_TRUE(addr != NULL);
    ASSERT_EQ(gmacSuccess, Mode_->memset(addr, 0x5a, Size_ * sizeof(int)));

    int *dst = new int[Size_];
    ASSERT_EQ(gmacSuccess, Mode_->copyToHost(dst, addr, Size_ * sizeof(int)));
    for(size_t i = 0; i < Size_; i++) ASSERT_EQ(0x5a5a5a5a, dst[i]);

    ASSERT_EQ(gmacSuccess, Mode_->free(addr));
    delete[] dst;
}
