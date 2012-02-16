#include "unit2/dsm/manager.h"

#include "hal/types.h"

#include "util/misc.h"

#include "gtest/gtest.h"

static __impl::hal::device *device = NULL;
static __impl::hal::aspace *as1    = NULL;
static manager *mgr = NULL;

__impl::hal::ptr::backend_type
manager_mapping_test::BASE_ADDR = 0x0000;

const int manager_mapping_test::MAP0_OFF = 0x0100;
const int manager_mapping_test::MAP1_OFF = 0x1000;
const int manager_mapping_test::MAP2_OFF = 0x2000;
const int manager_mapping_test::MAP3_OFF = 0x3000;
const int manager_mapping_test::MAP4_OFF = 0x4000;
const int manager_mapping_test::MAP5_OFF = 0x5000;

const size_t manager_mapping_test::MAP0_SIZE = 0x0100;
const size_t manager_mapping_test::MAP1_SIZE = 0x0200;
const size_t manager_mapping_test::MAP2_SIZE = 0x1000;
const size_t manager_mapping_test::MAP3_SIZE = 0x0900;
const size_t manager_mapping_test::MAP4_SIZE = 0x0700;
const size_t manager_mapping_test::MAP5_SIZE = 0x0800;

void
manager_mapping_test::SetUpTestCase()
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
    mgr = new manager();
    // Create address space
    as1 = device->create_aspace(__impl::hal::device::None, err);
    ASSERT_TRUE(err == gmacSuccess);
}

void manager_mapping_test::TearDownTestCase()
{
    gmacError_t err;
    err = device->destroy_aspace(*as1);
    ASSERT_TRUE(err == gmacSuccess);

    delete mgr;

    __impl::hal::fini();
}

TEST_F(manager_mapping_test, mappings_in_range)
{
    gmacError_t err;

    // We will use the same base allocation
    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), as1) + MAP0_OFF;
    ptr p1 = p0 + (MAP1_OFF - MAP0_OFF);
    ptr p2 = p0 + (MAP2_OFF - MAP0_OFF);
    ptr p3 = p0 + (MAP3_OFF - MAP0_OFF);
    ptr p4 = p0 + (MAP4_OFF - MAP0_OFF);
    ptr p5 = p0 + (MAP5_OFF - MAP0_OFF);

    mapping_ptr m0 = new mapping(p0);
    mapping_ptr m1 = new mapping(p1);
    mapping_ptr m2 = new mapping(p2);
    mapping_ptr m3 = new mapping(p3);
    mapping_ptr m4 = new mapping(p4);
    mapping_ptr m5 = new mapping(p5);

    block_ptr b0 = new block(MAP0_SIZE);
    block_ptr b1 = new block(MAP1_SIZE);
    block_ptr b2 = new block(MAP2_SIZE);
    block_ptr b3 = new block(MAP3_SIZE);
    block_ptr b4 = new block(MAP4_SIZE);
    block_ptr b5 = new block(MAP5_SIZE);

    err = m0->append(b0);
    ASSERT_TRUE(err == gmacSuccess);
    err = m1->append(b1);
    ASSERT_TRUE(err == gmacSuccess);
    err = m2->append(b2);
    ASSERT_TRUE(err == gmacSuccess);
    err = m3->append(b3);
    ASSERT_TRUE(err == gmacSuccess);
    err = m4->append(b4);
    ASSERT_TRUE(err == gmacSuccess);
    err = m5->append(b5);
    ASSERT_TRUE(err == gmacSuccess);

    err = mgr->helper_insert(*as1, m0);
    ASSERT_TRUE(err == gmacSuccess);
    err = mgr->helper_insert(*as1, m1);
    ASSERT_TRUE(err == gmacSuccess);
    err = mgr->helper_insert(*as1, m2);
    ASSERT_TRUE(err == gmacSuccess);
    err = mgr->helper_insert(*as1, m3);
    ASSERT_TRUE(err == gmacSuccess);
    err = mgr->helper_insert(*as1, m4);
    ASSERT_TRUE(err == gmacSuccess);
    err = mgr->helper_insert(*as1, m5);
    ASSERT_TRUE(err == gmacSuccess);

    delete b0;
    delete b1;
    delete b2;
    delete b3;
    delete b4;
    delete b5;

    delete m0;
    delete m1;
    delete m2;
    delete m3;
    delete m4;
    delete m5;
}
