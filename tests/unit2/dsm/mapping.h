#include "common.h"

#include "mock/hal/types.h"

#include "gtest/gtest.h"

#include "dsm/mapping.h"

#include "util/misc.h"

using I_UTIL::range;
using I_HAL::ptr;

using I_DSM::coherence::block;
using I_DSM::coherence::block_ptr;

class mapping_test : public testing::Test {
protected:
	static void SetUpTestCase();
	static void TearDownTestCase();
};

class mapping;

typedef mapping *mapping_ptr;

class GMAC_LOCAL mapping :
    public I_DSM::mapping
{
    typedef I_DSM::mapping parent;
public:
    typedef parent::pair_mapping pair_mapping;

    parent::range_block
    get_blocks_in_range(ptr::offset_type offset, size_t count);

    template <typename I>
    static mapping_ptr merge_mappings(range<I> range, ptr::offset_type off, size_t count);

    pair_mapping
    split(size_t off, size_t count, I_DSM::error &err)
    {
        return parent::split(off, count, err);
    }

    I_DSM::error
    prepend(block_ptr b)
    {
        return parent::prepend(b);
    }

    I_DSM::error
    append(block_ptr b)
    {
        return parent::append(b);
    }

    I_DSM::error
    append(mapping &&map)
    {
        return parent::append(std::move(map));
    }

    mapping(ptr addr, GmacProtection prot) :
        parent(addr, prot)
    {
    }

    I_DSM::error acquire(ptr::offset_type offset, size_t count, int flags);
    I_DSM::error release(ptr::offset_type offset, size_t count);

    template <typename I>
    static I_DSM::error link(ptr ptr1, range<I> range1, submappings &sub1,
                             ptr ptr2, range<I> range2, submappings &sub2, size_t count, int flags);

    //////////////////////
    // Helper functions //
    //////////////////////
    static block_ptr
    helper_create_block(size_t size)
    {
        return parent::factory_block::create(size);
    }

};
