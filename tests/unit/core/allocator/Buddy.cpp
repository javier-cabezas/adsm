#include "core/allocator/Buddy.h"

#include "gtest/gtest.h"

#include <list>

using gmac::core::allocator::Buddy;

class BuddyTest : public testing::Test {
public:
    static const size_t Size_ = 8 * 1024 * 1024;

    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {
    }
};


TEST_F(BuddyTest, SimpleAllocations) {
    Buddy buddy((void *)0x1000, Size_);
    ASSERT_TRUE(buddy.addr() == (void *)0x1000);

    std::list<void *> allocs;
    void *ret = 0;
    for(unsigned n = 2; n < 32; n = n * 2) {
        size_t size = Size_ / n;
        ret = buddy.get(size);
        ASSERT_EQ((unsigned long)0x1000, (unsigned long)ret % (Size_ / n));
        ASSERT_EQ(size, (Size_ / n));
        allocs.push_back(ret);
    }

    size_t size = Size_ / 2;
    ret = buddy.get(size);
    ASSERT_TRUE(ret == NULL);
    ASSERT_EQ(size, Size_ / 2);

    std::list<void *>::const_iterator i;
    unsigned n = 2;
    for(i = allocs.begin(); i != allocs.end(); i++) {
        buddy.put(*i, Size_ / n);
        n = n * 2;
    }
    allocs.clear();

    size = Size_ / 2;
    ret = buddy.get(size);
    ASSERT_TRUE(ret != NULL);
    ASSERT_EQ(size, Size_ / 2);
}
