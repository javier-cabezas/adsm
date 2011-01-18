#include "unit/init.h"

#include "core/IOBuffer.h"
#include "core/Mode.h"
#include "core/Process.h"

#include "gtest/gtest.h"

using __impl::core::IOBuffer;
using __impl::core::Mode;
using __impl::core::Process;

class IOBufferTest : public testing::Test {
public:
    static IOBuffer *Buffer_;
    static Mode *Mode_;

    const static size_t Size_ = 4 * 1024 * 1024;

    static void SetUpTestCase() {
        InitProcess();
    }

    static void TearDownTestCase() {
        Mode_->destroyIOBuffer(Buffer_);
        FiniProcess();
    }
};

IOBuffer *IOBufferTest::Buffer_ = NULL;
Mode *IOBufferTest::Mode_ = NULL;

TEST_F(IOBufferTest, Creation) {
    Mode &current = Mode::getCurrent();
    Mode_ = &current;
    ASSERT_TRUE(Mode_ != NULL);

    Buffer_ = current.createIOBuffer(Size_);
    ASSERT_TRUE(Buffer_ != NULL);
    ASSERT_TRUE(Buffer_->size() >= Size_);
}

TEST_F(IOBufferTest, ToAccelerator) {
    ASSERT_TRUE(memset(Buffer_->addr(), 0x7a, Buffer_->size()) == Buffer_->addr());

    accptr_t addr = NULL;
    ASSERT_EQ(gmacSuccess, Mode_->malloc(addr, Size_));
    ASSERT_EQ(gmacSuccess, Mode_->bufferToAccelerator(addr, *Buffer_, Size_));
    ASSERT_EQ(IOBuffer::ToAccelerator, Buffer_->state());

    ASSERT_EQ(gmacSuccess, Buffer_->wait());
    ASSERT_EQ(IOBuffer::Idle, Buffer_->state());

    int *dst = NULL;
    dst = new int[Size_ / sizeof(int)];
    ASSERT_TRUE(dst != NULL);
    ASSERT_EQ(gmacSuccess, Mode_->copyToHost(hostptr_t(dst), addr, Size_));
    for(size_t i = 0; i < Size_ / sizeof(int); i++) ASSERT_EQ(0x7a7a7a7a, dst[i]);

    ASSERT_EQ(gmacSuccess, Mode_->free(addr));    
    delete[] dst;
}

TEST_F(IOBufferTest, ToHost) {
    accptr_t addr = NULL;
    ASSERT_EQ(gmacSuccess, Mode_->malloc(addr, Size_));
    ASSERT_EQ(gmacSuccess, Mode_->memset(addr, 0x5b, Size_));
    
    ASSERT_EQ(gmacSuccess, Mode_->acceleratorToBuffer(*Buffer_, addr, Size_));
    ASSERT_EQ(IOBuffer::ToHost, Buffer_->state());

    ASSERT_EQ(gmacSuccess, Buffer_->wait());
    ASSERT_EQ(IOBuffer::Idle, Buffer_->state());

    int *ptr = reinterpret_cast<int *>(Buffer_->addr());
    for(size_t i = 0; i < Size_ / sizeof(int); i++) ASSERT_EQ(0x5b5b5b5b, ptr[i]);
    ASSERT_EQ(gmacSuccess, Mode_->free(addr));
}
