#include "unit/init.h"
#include "unit/core/hpe/Mode.h"

#include "core/hpe/Process.h"
#include "core/hpe/Mode.h"

#include "gtest/gtest.h"

using __impl::core::hpe::Mode;
using gmac::core::hpe::Process;

Mode *ModeTest::Mode_ = NULL;

void ModeTest::SetUpTestCase() {
    InitProcess();
    if(Mode_ != NULL) return;
	Process &proc = dynamic_cast<Process &>(Process::getInstance());
    Mode_ = dynamic_cast<Mode *>(proc.createMode(0));
    ASSERT_TRUE(Mode_ != NULL);
    Mode_->initThread();
}

void ModeTest::TearDownTestCase() {
    Mode_->detach();
    FiniProcess();
    Mode_ = NULL;
}


TEST_F(ModeTest, ModeCurrent) {
    Mode_->attach();
    Mode &current = Mode::getCurrent();
    ASSERT_TRUE(&current == Mode_);
}

TEST_F(ModeTest, ModeMemory) {
    hostptr_t fakePtr = (uint8_t *) 0xcafebabe;
    accptr_t addr(0);
    ASSERT_EQ(gmacSuccess, Mode_->map(addr, fakePtr, Size_));
    ASSERT_TRUE(addr != 0);

    ASSERT_EQ(gmacSuccess, Mode_->unmap(fakePtr, Size_));
}

TEST_F(ModeTest, MemoryCopy) {
    int *src = new int[Size_];
    ASSERT_TRUE(src != NULL);
    int *dst = new int[Size_];
    ASSERT_TRUE(dst != NULL);
    memset(src, 0x5a, Size_ * sizeof(int));
    accptr_t addr(0);
    ASSERT_EQ(gmacSuccess, Mode_->map(addr, hostptr_t(src), Size_ * sizeof(int)));
    ASSERT_TRUE(addr != 0);
    ASSERT_EQ(gmacSuccess, Mode_->copyToAccelerator(addr, hostptr_t(src), Size_ * sizeof(int)));
    ASSERT_EQ(gmacSuccess, Mode_->copyToHost(hostptr_t(dst), addr, Size_ * sizeof(int)));
    for(size_t i = 0; i < Size_; i++) ASSERT_EQ(0x5a5a5a5a, dst[i]);
    ASSERT_EQ(gmacSuccess, Mode_->unmap(hostptr_t(src), Size_ * sizeof(int)));
    delete[] dst;
    delete[] src;
}

TEST_F(ModeTest, MemorySet) {
    hostptr_t fakePtr = (uint8_t *) 0xcafebabe;
    accptr_t addr(0);
    ASSERT_EQ(gmacSuccess, Mode_->map(addr, fakePtr, Size_ * sizeof(int)));
    ASSERT_TRUE(addr != 0);
    ASSERT_EQ(gmacSuccess, Mode_->memset(addr, 0x5a, Size_ * sizeof(int)));

    int *dst = new int[Size_];
    ASSERT_EQ(gmacSuccess, Mode_->copyToHost(hostptr_t(dst), addr, Size_ * sizeof(int)));
    for(size_t i = 0; i < Size_; i++) ASSERT_EQ(0x5a5a5a5a, dst[i]);

    ASSERT_EQ(gmacSuccess, Mode_->unmap(fakePtr, Size_ * sizeof(int)));
    delete[] dst;
}
