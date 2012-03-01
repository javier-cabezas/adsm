#include "unit2/dsm/mapping.h"

#include "hal/types.h"

#include "util/misc.h"

#include "gtest/gtest.h"

static __impl::hal::device *device;
static __impl::hal::aspace *as1;
static __impl::hal::aspace *as2;

void
mapping_test::SetUpTestCase()
{
    // Inititalize platform
    gmacError_t err = __impl::hal::init();
    ASSERT_TRUE(err == gmacSuccess);
    // Get platforms
    __impl::hal::list_platform platforms = __impl::hal::get_platforms();
    ASSERT_TRUE(platforms.size() > 0);
    // Get devices
    __impl::hal::platform::list_device devices = platforms.front()->get_devices(__impl::hal::device::DEVICE_TYPE_GPU);
    ASSERT_TRUE(devices.size() > 0);
    device = devices.front();
    // Create address spaces
    as1 = device->create_aspace(__impl::hal::device::None, err);
    ASSERT_TRUE(err == gmacSuccess);
    as2 = device->create_aspace(__impl::hal::device::None, err);
    ASSERT_TRUE(err == gmacSuccess);
}

void mapping_test::TearDownTestCase()
{
    gmacError_t err;
    err = device->destroy_aspace(*as1);
    ASSERT_TRUE(err == gmacSuccess);
    err = device->destroy_aspace(*as2);
    ASSERT_TRUE(err == gmacSuccess);

    __impl::hal::fini();
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
    err = m0->append(m1);
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
    err = m0->append(m1);
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
    err = m0->append(m1);
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
    err = m0->append(m1);
    ASSERT_FALSE(err == error_dsm::DSM_SUCCESS);

    delete m0;
    delete m1;
}

