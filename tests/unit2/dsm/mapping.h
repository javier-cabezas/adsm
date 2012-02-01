#include "gtest/gtest.h"

#include "dsm/mapping.h"
#include "hal/types.h"
#include "util/misc.h"

using __impl::util::range;
using __impl::hal::ptr;

using __impl::dsm::coherence::block;
using __impl::dsm::coherence::block_ptr;

class mapping_test : public testing::Test {
protected:
	static void SetUpTestCase();
	static void TearDownTestCase();
};

class mapping;

typedef mapping *mapping_ptr;

class mapping :
    public __impl::dsm::mapping
{
    typedef __impl::dsm::mapping parent;
public:
    parent::range_block
    get_blocks_in_range(size_t offset, size_t count);

    template <typename I>
    static mapping_ptr merge_mappings(range<I> range, ptr::address addr, size_t count);

    static gmacError_t dup2(mapping_ptr map1, ptr::address addr1,
                            mapping_ptr map2, ptr::address addr2, size_t count);

    gmacError_t dup(ptr::address addr1, mapping_ptr map2, ptr::address addr2, size_t count);

    gmacError_t split(ptr::address addr, size_t count);

    gmacError_t prepend(block_ptr b)
    {
        return parent::prepend(b);
    }

    gmacError_t append(block_ptr b)
    {
        return parent::append(b);
    }

    gmacError_t append(mapping_ptr map);

    mapping(ptr addr) :
        parent(addr)
    {
    }

    gmacError_t acquire(size_t offset, size_t count, int flags);
    gmacError_t release(size_t offset, size_t count);

    template <typename I>
    static gmacError_t link(ptr ptr1, range<I> range1, submappings &sub1,
                            ptr ptr2, range<I> range2, submappings &sub2, size_t count, int flags);

};
