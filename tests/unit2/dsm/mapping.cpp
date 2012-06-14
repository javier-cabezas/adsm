#include "unit2/dsm/mapping.h"

#include "util/misc.h"

#include "gtest/gtest.h"

#include "hal/error.h"

static I_HAL::virt::aspace *as0;
static I_HAL::virt::aspace *as1;

void
mapping_test::SetUpTestCase()
{
    as0 = I_HAL::virt::aspace::create();
    as1 = I_HAL::virt::aspace::create();
}

void mapping_test::TearDownTestCase()
{
    I_HAL::virt::aspace::destroy(*as0);
    I_HAL::virt::aspace::destroy(*as1);
}

typedef __impl::util::bounds<ptr> alloc;

TEST_F(mapping_test, prepend_block)
{
    static const unsigned OFFSET = 0x400;

    I_HAL::virt::object_view view(*as0, 0);
    ptr ptrBase(view, 0);

    ptr p0 = ptrBase + OFFSET;

    mapping_ptr m0 = new mapping(p0 + OFFSET, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(0x100);
    block_ptr b1 = mapping::helper_create_block(0x300);

    I_DSM::error err;

    ASSERT_TRUE(m0->get_bounds().get_size() == 0);
    err = m0->prepend(b0);
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());
    err = m0->prepend(b1);
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() + b1->get_size()));

    delete m0;
}

TEST_F(mapping_test, prepend_block2)
{
    static const unsigned OFFSET = 0x400;

    I_HAL::virt::object_view view(*as0, 0);
    ptr ptrBase(view, 0);

    ptr p0 = ptrBase + OFFSET;

    mapping_ptr m0 = new mapping(p0, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(0x100);
    block_ptr b1 = mapping::helper_create_block(0x400);

    I_DSM::error err;

    ASSERT_TRUE(m0->get_bounds().get_size() == 0);
    err = m0->prepend(b0);
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());
    // The OFFSET 0x400 is not enough to fit the second block, TOTAL 0x500
    err = m0->prepend(b1);
    ASSERT_DSM_FAILURE(err);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());

    delete m0;
}

TEST_F(mapping_test, append_block)
{
    static const unsigned BLOCK_0_SIZE = 0x100;
    static const unsigned BLOCK_1_SIZE = 0x400;
    static const unsigned BLOCK_2_SIZE = 0x400;
    static const unsigned BLOCK_3_SIZE = 0x000;

    I_HAL::virt::object_view view(*as0, 0);
    ptr ptrBase(view, 0);

    ptr p0 = ptrBase;

    mapping_ptr m0 = new mapping(p0, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(BLOCK_0_SIZE);
    block_ptr b1 = mapping::helper_create_block(BLOCK_1_SIZE);
    block_ptr b2 = mapping::helper_create_block(BLOCK_2_SIZE);
    block_ptr b3 = mapping::helper_create_block(BLOCK_3_SIZE);

    I_DSM::error err;

    ASSERT_TRUE(m0->get_bounds().get_size() == 0);
    err = m0->append(b0);
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());
    err = m0->append(b1);
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size()));
    err = m0->append(b2);
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size() +
                                                b2->get_size()));
    err = m0->append(b3);
    // Blocks of size 0 are not allowed
    ASSERT_DSM_FAILURE(err);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size() +
                                                b2->get_size()));

    delete m0;
}

TEST_F(mapping_test, append_mapping)
{
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE + 0x100;

    I_HAL::virt::object_view view(*as0, 0);
    ptr ptrBase(view, 0);

    ptr p0 = ptrBase;
    ptr p1 = p0 + OFFSET;

    mapping_ptr m0 = new mapping(p0, GMAC_PROT_READWRITE);
    mapping_ptr m1 = new mapping(p1, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);
    block_ptr b1 = mapping::helper_create_block(BLOCK_SIZE);

    I_DSM::error err;

    err = m0->append(b0);
    ASSERT_DSM_SUCCESS(err);
    err = m1->append(b1);
    ASSERT_DSM_SUCCESS(err);
    err = m0->append(std::move(*m1));
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size() +
                                                (OFFSET - b0->get_size())));

    delete m0;
    delete m1;
}

TEST_F(mapping_test, append_mapping2)
{
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE - 0x100;

    I_HAL::virt::object_view view(*as0, 0);
    ptr ptrBase(view, 0);

    ptr p0 = ptrBase;
    ptr p1 = p0 + OFFSET;

    mapping_ptr m0 = new mapping(p0, GMAC_PROT_READWRITE);
    mapping_ptr m1 = new mapping(p1, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);
    block_ptr b1 = mapping::helper_create_block(BLOCK_SIZE);

    I_DSM::error err;

    err = m0->append(b0);
    ASSERT_DSM_SUCCESS(err);
    err = m1->append(b1);
    ASSERT_DSM_SUCCESS(err);
    err = m0->append(std::move(*m1));
    ASSERT_DSM_FAILURE(err);

    delete m0;
    delete m1;
}

TEST_F(mapping_test, append_mapping3)
{
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE;

    I_HAL::virt::object_view view(*as0, 0);
    ptr ptrBase(view, 0);

    ptr p0 = ptrBase;
    ptr p1 = p0 + OFFSET;

    mapping_ptr m0 = new mapping(p0, GMAC_PROT_READWRITE);
    mapping_ptr m1 = new mapping(p1, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);
    block_ptr b1 = mapping::helper_create_block(BLOCK_SIZE);

    I_DSM::error err;

    err = m0->append(b0);
    ASSERT_DSM_SUCCESS(err);
    err = m1->append(b1);
    ASSERT_DSM_SUCCESS(err);
    err = m0->append(std::move(*m1));
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size()));

    delete m0;
    delete m1;
}

TEST_F(mapping_test, append_mapping4)
{
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE;

    I_HAL::virt::object_view view0(*as0, 0);
    I_HAL::virt::object_view view1(*as1, 0);
    ptr p0(view0, 0);
    ptr p1(view1, 0);

    mapping_ptr m0 = new mapping(p0,          GMAC_PROT_READWRITE);
    mapping_ptr m1 = new mapping(p1 + OFFSET, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);
    block_ptr b1 = mapping::helper_create_block(BLOCK_SIZE);

    I_DSM::error err;

    err = m0->append(b0);
    ASSERT_DSM_SUCCESS(err);
    err = m1->append(b1);
    ASSERT_DSM_SUCCESS(err);
    err = m0->append(std::move(*m1));
    ASSERT_DSM_FAILURE(err);

    delete m0;
    delete m1;
}

TEST_F(mapping_test, split)
{
    static const unsigned BLOCK_SIZE = 0x10000;

    I_HAL::virt::object_view view(*as0, 0);
    ptr p0(view, 0);

    mapping_ptr m0 = new mapping(p0, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);

    I_DSM::error err;

    err = m0->append(b0);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);

    mapping::pair_mapping pair = m0->split(0x3500, 0x3000, err);
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(pair.first != nullptr && pair.second != nullptr);

    ASSERT_TRUE(m0->get_nblocks() == 1);
    ASSERT_TRUE(pair.first->get_nblocks() == 1);
    ASSERT_TRUE(pair.second->get_nblocks() == 1);

    ASSERT_TRUE((m0->get_bounds().get_size() + 
                 pair.first->get_bounds().get_size() + 
                 pair.second->get_bounds().get_size()) == BLOCK_SIZE);

    delete m0;

    delete pair.first;
    delete pair.second;
}

TEST_F(mapping_test, split2)
{
    static const unsigned BLOCK_SIZE = 0x10000;

    I_HAL::virt::object_view view(*as0, 0);
    ptr p0(view, 0);

    mapping_ptr m0 = new mapping(p0, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);

    I_DSM::error err;

    err = m0->append(b0);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);

    mapping::pair_mapping pair = m0->split(0, 0x3000, err);
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(pair.first != nullptr);
    ASSERT_TRUE(pair.second == nullptr);

    ASSERT_TRUE(m0->get_nblocks() == 1);
    ASSERT_TRUE(pair.first->get_nblocks() == 1);

    ASSERT_TRUE((m0->get_bounds().get_size() + 
                 pair.first->get_bounds().get_size()) == BLOCK_SIZE);

    delete m0;

    delete pair.first;
}

TEST_F(mapping_test, split3)
{
    static const unsigned BLOCK_SIZE = 0x10000;

    I_HAL::virt::object_view view(*as0, 0);
    ptr p0(view, 0);

    mapping_ptr m0 = new mapping(p0, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);

    I_DSM::error err;

    err = m0->append(b0);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);

    mapping::pair_mapping pair = m0->split(BLOCK_SIZE - 0x3000, 0x3000, err);
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(pair.first != nullptr);
    ASSERT_TRUE(pair.second == nullptr);

    ASSERT_TRUE(m0->get_nblocks() == 1);
    ASSERT_TRUE(pair.first->get_nblocks() == 1);

    ASSERT_TRUE((m0->get_bounds().get_size() + 
                 pair.first->get_bounds().get_size()) == BLOCK_SIZE);

    delete m0;

    delete pair.first;
}

TEST_F(mapping_test, split_linked)
{
    static const unsigned BLOCK_SIZE = 0x10000;

    I_HAL::virt::object_view view0(*as0, 0);
    I_HAL::virt::object_view view1(*as1, 0);
    ptr p0(view0, 0);
    ptr p1(view0, 0);

    mapping_ptr m0 = new mapping(p0, GMAC_PROT_READWRITE);
    mapping_ptr m1 = new mapping(p1, GMAC_PROT_READWRITE);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);

    I_DSM::error err;

    err = m0->append(b0);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m1->append(b0);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);

    mapping::pair_mapping pair = m0->split(0x3500, 0x3000, err);
    ASSERT_DSM_SUCCESS(err);
    ASSERT_TRUE(pair.first != nullptr);
    ASSERT_TRUE(pair.second != nullptr);

    ASSERT_TRUE(m0->get_nblocks() == 1);
    ASSERT_TRUE(pair.first->get_nblocks() == 1);
    ASSERT_TRUE(pair.second->get_nblocks() == 1);
    // Splitting should be reflected in the other mapping
    ASSERT_TRUE(m1->get_nblocks() == 3);

    ASSERT_TRUE((m0->get_bounds().get_size() + 
    		     pair.first->get_bounds().get_size() +
                 pair.second->get_bounds().get_size()) == BLOCK_SIZE);

    delete m0;
    delete m1;

    delete pair.first;
    delete pair.second;
}
