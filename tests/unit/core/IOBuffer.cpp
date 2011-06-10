#include "unit/core/IOBuffer.h"

#include "core/IOBuffer.h"

#include "gtest/gtest.h"

using __impl::core::IOBuffer;
using __impl::core::Mode;


void IOBufferTest::ToAccelerator(Mode &mode)
{
    IOBuffer &buffer = mode.createIOBuffer(Size_, GMAC_PROT_WRITE);
    ASSERT_TRUE(buffer.size() >= Size_);

    ASSERT_TRUE(memset(buffer.addr(), 0x7a, buffer.size()) == buffer.addr());

    hostptr_t fakePtr = (uint8_t *) 0xcafebabe;
    accptr_t addr(0);
    ASSERT_EQ(gmacSuccess, mode.map(addr, fakePtr, Size_));

    ASSERT_EQ(gmacSuccess, mode.bufferToAccelerator(addr, buffer, Size_));

    ASSERT_EQ(gmacSuccess, buffer.wait());
    ASSERT_EQ(IOBuffer::Idle, buffer.state());

    int *dst = NULL;
    dst = new int[Size_ / sizeof(int)];
    ASSERT_TRUE(dst != NULL);
    ASSERT_EQ(gmacSuccess, mode.copyToHost(hostptr_t(dst), addr, Size_));
    for(size_t i = 0; i < Size_ / sizeof(int); i++) ASSERT_EQ(0x7a7a7a7a, dst[i]);

    ASSERT_EQ(gmacSuccess, mode.unmap(fakePtr, Size_));
    delete[] dst;

    mode.destroyIOBuffer(buffer);
}

void IOBufferTest::ToHost(Mode &mode)
{
    IOBuffer &buffer = mode.createIOBuffer(Size_, GMAC_PROT_READ);
    ASSERT_TRUE(buffer.size() >= Size_);

    hostptr_t fakePtr = (uint8_t *) 0xcafebabe;
    accptr_t addr(0);
    ASSERT_EQ(gmacSuccess, mode.map(addr, fakePtr, Size_));
    ASSERT_EQ(gmacSuccess, mode.memset(addr, 0x5b, Size_));
    
    ASSERT_EQ(gmacSuccess, mode.acceleratorToBuffer(buffer, addr, Size_));

    ASSERT_EQ(gmacSuccess, buffer.wait());
    ASSERT_EQ(IOBuffer::Idle, buffer.state());

    int *ptr = reinterpret_cast<int *>(buffer.addr());
    for(size_t i = 0; i < Size_ / sizeof(int); i++) ASSERT_EQ(0x5b5b5b5b, ptr[i]);
    ASSERT_EQ(gmacSuccess, mode.unmap(fakePtr, Size_));

    mode.destroyIOBuffer(buffer);
}
