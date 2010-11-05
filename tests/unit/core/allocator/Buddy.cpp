#include "core/allocator/Buddy.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <list>

using gmac::core::allocator::Buddy;

class BuddyTest : public testing::Test {
public:
    static const size_t Size_ = 8 * 1024 * 1024;
    static unsigned long Base_;
    static void *BasePtr_;

    static void SetUpTestCase() {
    }

    static void TearDownTestCase() {
    }
};


unsigned long BuddyTest::Base_ = 0x1000;
void *BuddyTest::BasePtr_ = (void *)Base_;

TEST_F(BuddyTest, SimpleAllocations) {
    Buddy buddy(BasePtr_, Size_);
    ASSERT_TRUE(buddy.addr() == BasePtr_);

    std::list<void *> allocs;
    void *ret = 0;
    for(unsigned n = 2; n < 32; n = n * 2) {
        size_t size = Size_ / n;
        ret = buddy.get(size);
        ASSERT_EQ(Base_, (unsigned long)ret % (Size_ / n));
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

TEST_F(BuddyTest, RandomAllocations) {
    const int Allocations = 1024;
    const size_t MinMemory = 4 * 4096;
    size_t freeMemory = Size_;
    Buddy buddy(BasePtr_, Size_);
    ASSERT_TRUE(buddy.addr() == BasePtr_);

    typedef std::map<void *, size_t> AllocMap;
    AllocMap map;
    srand(time(NULL));
    for(int i = 0; i < Allocations; i++) {
        // Generate a random size to be allocated
        size_t s = 0;
        while(s == 0) s = size_t(freeMemory * rand() / (RAND_MAX + 1.0));

        // Perform the allocation
        size_t check = s;
        void *addr = buddy.get(s);
        if(addr == NULL) {--i; continue; } // Too much memory, try again
        ASSERT_GE(s, check);
        std::pair<AllocMap::iterator, bool> p = map.insert(AllocMap::value_type(addr, s));
        ASSERT_EQ(true, p.second);
        freeMemory -= s;

        if(freeMemory > MinMemory) continue;

        int n = int(map.size() * rand() / (RAND_MAX + 1.0));
        AllocMap::iterator i = map.begin();
        for(; n > 0; n--, i++);
        buddy.put(i->first, i->second);
        freeMemory += i->second;
        map.erase(i);
    }

}
