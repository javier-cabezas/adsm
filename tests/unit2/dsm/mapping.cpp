#include "unit2/dsm/mapping.h"

#include "hal/types.h"

#include "util/misc.h"

#include "gtest/gtest.h"

namespace I_HAL = __impl::hal;

static I_HAL::phys::list_platform platforms;
static I_HAL::phys::platform *plat0;
static I_HAL::phys::platform::set_memory memories;
static I_HAL::phys::platform::set_processing_unit pUnits;
static I_HAL::phys::processing_unit *pUnit0;
static I_HAL::phys::aspace *pas0;
static I_HAL::virt::aspace *as0;
static I_HAL::virt::aspace *as1;

static I_HAL::virt::object *obj0;

void
mapping_test::SetUpTestCase()
{
    // Inititalize platform
    gmacError_t err = I_HAL::init();
    ASSERT_TRUE(err == gmacSuccess);
    // Get platforms
    platforms = I_HAL::phys::get_platforms();
    ASSERT_TRUE(platforms.size() > 0);
    plat0 = platforms.front();
    // Get processing units
    pUnits = plat0->get_processing_units(I_HAL::phys::processing_unit::PUNIT_TYPE_GPU);
    ASSERT_TRUE(pUnits.size() > 0);
    pUnit0 = *pUnits.begin();
    pas0 = &pUnit0->get_paspace();

    // Create address spaces
    I_HAL::phys::aspace::set_processing_unit compatibleUnits({ pUnit0 });
    as0 = pas0->create_vaspace(compatibleUnits, err);
    ASSERT_TRUE(err == gmacSuccess);
    as1 = pas0->create_vaspace(compatibleUnits, err);
    ASSERT_TRUE(err == gmacSuccess);
}

void mapping_test::TearDownTestCase()
{
    gmacError_t err;

    err = pas0->destroy_vaspace(*as0);
    ASSERT_TRUE(err == gmacSuccess);
    err = pas0->destroy_vaspace(*as1);
    ASSERT_TRUE(err == gmacSuccess);

    I_HAL::fini();
}

typedef __impl::util::bounds<ptr> alloc;

TEST_F(mapping_test, prepend_block)
{
    static const unsigned OFFSET = 0x400;

    gmacError_t errHal;

    obj0 = plat0->create_object(**pas0->get_memories().begin(), OFFSET, errHal);

    ptr ptrBase = as0->map(*obj0, errHal);
    ASSERT_TRUE(errHal == gmacSuccess);

    ptr p0 = ptrBase + OFFSET;

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

    errHal = as0->unmap(ptrBase);
    ASSERT_TRUE(errHal == gmacSuccess);
    errHal = plat0->destroy_object(*obj0);
    ASSERT_TRUE(errHal == gmacSuccess);
}

TEST_F(mapping_test, prepend_block2)
{
    static const unsigned OFFSET = 0x400;

    gmacError_t errHal;

    obj0 = plat0->create_object(**pas0->get_memories().begin(), OFFSET, errHal);

    ptr ptrBase = as0->map(*obj0, errHal);
    ASSERT_TRUE(errHal == gmacSuccess);

    ptr p0 = ptrBase + OFFSET;

    mapping_ptr m0 = new mapping(p0);

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

    errHal = as0->unmap(ptrBase);
    ASSERT_TRUE(errHal == gmacSuccess);
    errHal = plat0->destroy_object(*obj0);
    ASSERT_TRUE(errHal == gmacSuccess);
}

TEST_F(mapping_test, append_block)
{
    static const unsigned BLOCK_0_SIZE = 0x100;
    static const unsigned BLOCK_1_SIZE = 0x400;
    static const unsigned BLOCK_2_SIZE = 0x400;
    static const unsigned BLOCK_3_SIZE = 0x000;

    gmacError_t errHal;

    obj0 = plat0->create_object(**pas0->get_memories().begin(), BLOCK_0_SIZE +
                                                                BLOCK_1_SIZE +
                                                                BLOCK_2_SIZE +
                                                                BLOCK_3_SIZE, errHal);

    ptr ptrBase = as0->map(*obj0, errHal);
    ASSERT_TRUE(errHal == gmacSuccess);

    ptr p0 = ptrBase;

    mapping_ptr m0 = new mapping(p0);

    block_ptr b0 = mapping::helper_create_block(BLOCK_0_SIZE);
    block_ptr b1 = mapping::helper_create_block(BLOCK_1_SIZE);
    block_ptr b2 = mapping::helper_create_block(BLOCK_2_SIZE);
    block_ptr b3 = mapping::helper_create_block(BLOCK_3_SIZE);

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

    errHal = as0->unmap(ptrBase);
    ASSERT_TRUE(errHal == gmacSuccess);
    errHal = plat0->destroy_object(*obj0);
    ASSERT_TRUE(errHal == gmacSuccess);
}

TEST_F(mapping_test, append_mapping)
{
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE + 0x100;

    gmacError_t errHal;

    obj0 = plat0->create_object(**pas0->get_memories().begin(), BLOCK_SIZE +
                                                                BLOCK_SIZE, errHal);

    ptr ptrBase = as0->map(*obj0, errHal);
    ASSERT_TRUE(errHal == gmacSuccess);

    ptr p0 = ptrBase;
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

    errHal = as0->unmap(ptrBase);
    ASSERT_TRUE(errHal == gmacSuccess);
    errHal = plat0->destroy_object(*obj0);
    ASSERT_TRUE(errHal == gmacSuccess);
}

TEST_F(mapping_test, append_mapping2)
{
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE - 0x100;

    gmacError_t errHal;

    obj0 = plat0->create_object(**pas0->get_memories().begin(), BLOCK_SIZE +
                                                                BLOCK_SIZE, errHal);

    ptr ptrBase = as0->map(*obj0, errHal);
    ASSERT_TRUE(errHal == gmacSuccess);

    ptr p0 = ptrBase;
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

    errHal = as0->unmap(ptrBase);
    ASSERT_TRUE(errHal == gmacSuccess);
    errHal = plat0->destroy_object(*obj0);
    ASSERT_TRUE(errHal == gmacSuccess);
}

TEST_F(mapping_test, append_mapping3)
{
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE;

    gmacError_t errHal;

    obj0 = plat0->create_object(**pas0->get_memories().begin(), BLOCK_SIZE +
                                                                BLOCK_SIZE, errHal);

    ptr ptrBase = as0->map(*obj0, errHal);
    ASSERT_TRUE(errHal == gmacSuccess);

    ptr p0 = ptrBase;
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

    errHal = as0->unmap(ptrBase);
    ASSERT_TRUE(errHal == gmacSuccess);
    errHal = plat0->destroy_object(*obj0);
    ASSERT_TRUE(errHal == gmacSuccess);
}

TEST_F(mapping_test, append_mapping4)
{
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE;

    gmacError_t errHal;

    obj0 = plat0->create_object(**pas0->get_memories().begin(), BLOCK_SIZE +
                                                                BLOCK_SIZE, errHal);

    ptr p0 = as0->map(*obj0, errHal);
    ASSERT_TRUE(errHal == gmacSuccess);
    ptr p1 = as1->map(*obj0, errHal);
    ASSERT_TRUE(errHal == gmacSuccess);

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

    errHal = as0->unmap(p0);
    ASSERT_TRUE(errHal == gmacSuccess);
    errHal = as1->unmap(p1);
    ASSERT_TRUE(errHal == gmacSuccess);
    errHal = plat0->destroy_object(*obj0);
    ASSERT_TRUE(errHal == gmacSuccess);
}

