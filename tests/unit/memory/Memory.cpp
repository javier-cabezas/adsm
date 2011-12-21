#include "gtest/gtest.h"
#include "memory/Memory.h"

using __impl::memory::Memory;

static const int Size_ = 1024 * 1024;
static const int Pattern_ = 0xa5a5a5a5;


TEST(MemoryTest, MemoryShadowing) {
    int *addr_ = NULL;
    int *shadow_ = NULL;

    addr_ = (int *)memory_ops::map(NULL, Size_ * sizeof(int), GMAC_PROT_READWRITE);
    ASSERT_TRUE(addr_ != NULL);

    shadow_ = (int *)memory_ops::shadow(host_ptr(addr_), Size_ * sizeof(int));
    ASSERT_TRUE(shadow_ != NULL);
    ASSERT_TRUE(addr_ != shadow_);

    for(int n = 0; n < Size_; n++) {
        addr_[n] = n;
        ASSERT_TRUE(shadow_[n] == n);
    }

    for(int n = 0; n < Size_; n += 2) {
        shadow_[n] = Pattern_;
        ASSERT_TRUE(addr_[n] == Pattern_);
        ASSERT_TRUE(addr_[n + 1] == n + 1);
    }

    memory_ops::unshadow(host_ptr(shadow_), Size_ * sizeof(int));
    memory_ops::unmap(host_ptr(addr_), Size_ * sizeof(int));
}

