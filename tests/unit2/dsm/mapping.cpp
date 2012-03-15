#include "unit2/dsm/mapping.h"

#include "hal/types.h"

#include "util/misc.h"

#include "gtest/gtest.h"

namespace I_HAL = __impl::hal;

static I_HAL::phys::processing_unit *pUnit;
static I_HAL::phys::platform::set_aspace::value_type pas1;
static I_HAL::virt::aspace *as1;
static I_HAL::virt::aspace *as2;

void
mapping_test::SetUpTestCase()
{
    // Inititalize platform
    gmacError_t err = I_HAL::init();
    ASSERT_TRUE(err == gmacSuccess);
    // Get platforms
    I_HAL::phys::list_platform platforms = I_HAL::phys::get_platforms();
    ASSERT_TRUE(platforms.size() > 0);
    // Get processing units
    I_HAL::phys::platform::set_processing_unit pUnits = platforms.front()->get_processing_units(I_HAL::phys::processing_unit::PUNIT_TYPE_GPU);
    ASSERT_TRUE(pUnits.size() > 0);
    pUnit = *pUnits.begin();
    I_HAL::phys::platform::set_aspace pAspaces = pUnit->get_paspaces();
    pas1 = *pAspaces.begin();
    // Create address spaces
    as1 = pas1->create_vaspace(err);
    ASSERT_TRUE(err == gmacSuccess);
    as2 = pas1->create_vaspace(err);
    ASSERT_TRUE(err == gmacSuccess);
}

void mapping_test::TearDownTestCase()
{
    gmacError_t err;
    err = pas1->destroy_vaspace(*as1);
    ASSERT_TRUE(err == gmacSuccess);
    err = pas1->destroy_vaspace(*as2);
    ASSERT_TRUE(err == gmacSuccess);

    I_HAL::fini();
}

typedef __impl::util::bounds<ptr> alloc;

TEST_F(mapping_test, prepend_block)
{
    static const unsigned BASE_ADDR = 0x100;
    static const unsigned OFFSET    = 0x400;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), as1);

    mapping_ptr m0 = new mapping(p0 + OFFSET);

    block_ptr b0 = mapping::helper_create_block(0x100);
    block_ptr b1 = mapping::helper_create_block(0x300);

    error_dsm err;

    ASSERT_TRUE(m0->get_bounds().get_size() == 0);
    err = m0->prepend(b0);
    ASSERT_TRUE(err == __impl::dsm::DSM_SUCCESS);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());
    err = m0->prepend(b1);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() + b1->get_size()));

    delete m0;
}

TEST_F(mapping_test, prepend_block2)
{
    static const unsigned BASE_ADDR = 0x100;
    static const unsigned OFFSET    = 0x400;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), as1);

    mapping_ptr m0 = new mapping(p0 + OFFSET);

    block_ptr b0 = mapping::helper_create_block(0x100);
    block_ptr b1 = mapping::helper_create_block(0x400);

    error_dsm err;

    ASSERT_TRUE(m0->get_bounds().get_size() == 0);
    err = m0->prepend(b0);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());
    // The OFFSET 0x400 is not enough to fit the second block, TOTAL 0x500
    err = m0->prepend(b1);
    ASSERT_FALSE(err == error_dsm::DSM_SUCCESS);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());

    delete m0;
}

TEST_F(mapping_test, append_block)
{
    static const unsigned BASE_ADDR = 0x100;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), as1);

    mapping_ptr m0 = new mapping(p0);

    block_ptr b0 = mapping::helper_create_block(0x100);
    block_ptr b1 = mapping::helper_create_block(0x400);
    block_ptr b2 = mapping::helper_create_block(0x400);
    block_ptr b3 = mapping::helper_create_block(0x000);

    error_dsm err;

    ASSERT_TRUE(m0->get_bounds().get_size() == 0);
    err = m0->append(b0);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());
    err = m0->append(b1);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size()));
    err = m0->append(b2);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size() +
                                                b2->get_size()));
    err = m0->append(b3);
    // Blocks of size 0 are not allowed
    ASSERT_FALSE(err == error_dsm::DSM_SUCCESS);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size() +
                                                b2->get_size()));

    delete m0;
}

TEST_F(mapping_test, append_mapping)
{
    static const unsigned BASE_ADDR  = 0x100;
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE + 0x100;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), as1);
    ptr p1 = p0 + OFFSET;

    mapping_ptr m0 = new mapping(p0);
    mapping_ptr m1 = new mapping(p1);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);
    block_ptr b1 = mapping::helper_create_block(BLOCK_SIZE);

    error_dsm err;

    err = m0->append(b0);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    err = m1->append(b1);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    err = m0->append(std::move(*m1));
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size() +
                                                (OFFSET - b0->get_size())));

    delete m0;
    delete m1;
}

TEST_F(mapping_test, append_mapping2)
{
    static const unsigned BASE_ADDR  = 0x100;
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE - 0x100;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), as1);
    ptr p1 = p0 + OFFSET;

    mapping_ptr m0 = new mapping(p0);
    mapping_ptr m1 = new mapping(p1);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);
    block_ptr b1 = mapping::helper_create_block(BLOCK_SIZE);

    error_dsm err;

    err = m0->append(b0);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    err = m1->append(b1);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    err = m0->append(std::move(*m1));
    ASSERT_FALSE(err == error_dsm::DSM_SUCCESS);

    delete m0;
    delete m1;
}

TEST_F(mapping_test, append_mapping3)
{
    static const unsigned BASE_ADDR  = 0x100;
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), as1);
    ptr p1 = p0 + OFFSET;

    mapping_ptr m0 = new mapping(p0);
    mapping_ptr m1 = new mapping(p1);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);
    block_ptr b1 = mapping::helper_create_block(BLOCK_SIZE);

    error_dsm err;

    err = m0->append(b0);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    err = m1->append(b1);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    err = m0->append(std::move(*m1));
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size()));

    delete m0;
    delete m1;
}

TEST_F(mapping_test, append_mapping4)
{
    static const unsigned BASE_ADDR  = 0x100;
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), as1);
    ptr p1 = ptr(ptr::backend_ptr(BASE_ADDR), as2);

    mapping_ptr m0 = new mapping(p0);
    mapping_ptr m1 = new mapping(p1 + OFFSET);

    block_ptr b0 = mapping::helper_create_block(BLOCK_SIZE);
    block_ptr b1 = mapping::helper_create_block(BLOCK_SIZE);

    error_dsm err;

    err = m0->append(b0);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    err = m1->append(b1);
    ASSERT_TRUE(err == error_dsm::DSM_SUCCESS);
    err = m0->append(std::move(*m1));
    ASSERT_FALSE(err == error_dsm::DSM_SUCCESS);

    delete m0;
    delete m1;
}

