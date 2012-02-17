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

static ptr         p0, p1, p2, p3, p4, p5;
static mapping_ptr m0, m1, m2, m3, m4, m5;
static block_ptr   b0, b1, b2, b3, b4, b5;

static void range_init()
{
    // We use the same base allocation
    p0 = ptr(ptr::backend_ptr(manager_mapping_test::BASE_ADDR), as1) + manager_mapping_test::MAP0_OFF;
    p1 = p0 + (manager_mapping_test::MAP1_OFF - manager_mapping_test::MAP0_OFF);
    p2 = p0 + (manager_mapping_test::MAP2_OFF - manager_mapping_test::MAP0_OFF);
    p3 = p0 + (manager_mapping_test::MAP3_OFF - manager_mapping_test::MAP0_OFF);
    p4 = p0 + (manager_mapping_test::MAP4_OFF - manager_mapping_test::MAP0_OFF);
    p5 = p0 + (manager_mapping_test::MAP5_OFF - manager_mapping_test::MAP0_OFF);

    m0 = new mapping(p0);
    m1 = new mapping(p1);
    m2 = new mapping(p2);
    m3 = new mapping(p3);
    m4 = new mapping(p4);
    m5 = new mapping(p5);

    b0 = new block(manager_mapping_test::MAP0_SIZE);
    b1 = new block(manager_mapping_test::MAP1_SIZE);
    b2 = new block(manager_mapping_test::MAP2_SIZE);
    b3 = new block(manager_mapping_test::MAP3_SIZE);
    b4 = new block(manager_mapping_test::MAP4_SIZE);
    b5 = new block(manager_mapping_test::MAP5_SIZE);

    gmacError_t err;
    bool berr;

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

    berr = mgr->helper_insert(*as1, m0);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as1, m1);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as1, m2);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as1, m3);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as1, m4);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as1, m5);
    ASSERT_TRUE(berr);
}

static void range_fini()
{
    bool berr;

    berr = mgr->helper_clear_mappings(*as1);
    ASSERT_TRUE(berr);

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

template <typename T>
T
get_last_in_range(range<T> &r)
{
    for (typename range<T>::iterator it =  r.begin;
                                     it != r.end;
                                   ++it) {
        typename range<T>::iterator it2 = it;
        ++it2;
        if (it2 == r.end) return it;
    }
    return r.end;
}


TEST_F(manager_mapping_test, mappings_in_range)
{
    range_init();

    manager::map_mapping_group &group = mgr->get_aspace_mappings(*as1);

    // CASE1
    // p0, p0 + (p5 - p0): [p0, p4]
    //
    manager::range_mapping range = mgr->get_mappings_in_range(group, p0,
                                                              manager_mapping_test::MAP5_OFF -
                                                              manager_mapping_test::MAP0_OFF);
    ASSERT_TRUE((*range.begin)->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range))->get_ptr() == p4);

    // CASE2
    // p0, p0 + (p5 - p0) + 1: [p0, p5] 
    //
    manager::range_mapping range2 = mgr->get_mappings_in_range(group, p0,
                                                               manager_mapping_test::MAP5_OFF -
                                                               manager_mapping_test::MAP0_OFF + 1);
    ASSERT_TRUE((*range2.begin)->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range2))->get_ptr() == p5);

    // CASE3
    // p0 + size0, p0 + (p5 - p0): [p1, p4] 
    //
    manager::range_mapping range3 = mgr->get_mappings_in_range(group, p0 + manager_mapping_test::MAP0_SIZE,
                                                               (manager_mapping_test::MAP5_OFF -
                                                                manager_mapping_test::MAP0_OFF) -
                                                               manager_mapping_test::MAP0_SIZE);
    ASSERT_TRUE((*range3.begin)->get_ptr() == p1);
    ASSERT_TRUE((*get_last_in_range(range3))->get_ptr() == p4);

    // CASE4
    // p0 + size0 - 1, p0 + (p5 - p0): [p0, p4] 
    //
    manager::range_mapping range4 = mgr->get_mappings_in_range(group, p0 + (manager_mapping_test::MAP0_SIZE - 1),
                                                               (manager_mapping_test::MAP5_OFF -
                                                                manager_mapping_test::MAP0_OFF) -
                                                                manager_mapping_test::MAP0_SIZE);
    ASSERT_TRUE((*range4.begin)->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range4))->get_ptr() == p4);

    // CASE5
    // p3, p3 + size3: [p3, p3] 
    //
    manager::range_mapping range5 = mgr->get_mappings_in_range(group, p3,
                                                               manager_mapping_test::MAP3_SIZE);
    ASSERT_TRUE((*range5.begin)->get_ptr() == p3);
    ASSERT_TRUE((*get_last_in_range(range5))->get_ptr() == p3);


    // CASE6
    // p3 - 1, p3 + size3 - 1: [p2, p3] 
    //
    manager::range_mapping range6 = mgr->get_mappings_in_range(group, p3 - 1,
                                                               manager_mapping_test::MAP3_SIZE);
    ASSERT_TRUE((*range6.begin)->get_ptr() == p2);
    ASSERT_TRUE((*get_last_in_range(range6))->get_ptr() == p3);

    // CASE7
    // p4 - 1, p4 + size4 - 1: [p4, p4] 
    //
    manager::range_mapping range7 = mgr->get_mappings_in_range(group, p4 - 1,
                                                               manager_mapping_test::MAP4_SIZE);
    ASSERT_TRUE((*range7.begin)->get_ptr() == p4);
    ASSERT_TRUE((*get_last_in_range(range7))->get_ptr() == p4);

    
    range_fini();
}
