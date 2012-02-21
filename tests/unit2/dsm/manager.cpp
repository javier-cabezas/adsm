#include "unit2/dsm/manager.h"

#include "hal/types.h"

#include "util/misc.h"

#include "gtest/gtest.h"

static __impl::hal::device *device = NULL;
static __impl::hal::aspace *as1    = NULL;
static manager *mgr = NULL;

static ptr::backend_type BASE_ADDR = 0x0000;

static const int MAP0_OFF = 0x0100;
static const int MAP1_OFF = 0x1000;
static const int MAP2_OFF = 0x2000;
static const int MAP3_OFF = 0x3000;
static const int MAP4_OFF = 0x4000;
static const int MAP5_OFF = 0x5000;

static const size_t MAP0_SIZE = 0x0100;
static const size_t MAP1_SIZE = 0x0200;
static const size_t MAP2_SIZE = 0x1000;
static const size_t MAP3_SIZE = 0x0900;
static const size_t MAP4_SIZE = 0x0700;
static const size_t MAP5_SIZE = 0x0800;

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
    p0 = ptr(ptr::backend_ptr(BASE_ADDR), as1) + MAP0_OFF;
    p1 = p0 + (MAP1_OFF - MAP0_OFF);
    p2 = p0 + (MAP2_OFF - MAP0_OFF);
    p3 = p0 + (MAP3_OFF - MAP0_OFF);
    p4 = p0 + (MAP4_OFF - MAP0_OFF);
    p5 = p0 + (MAP5_OFF - MAP0_OFF);

    m0 = new mapping(p0);
    m1 = new mapping(p1);
    m2 = new mapping(p2);
    m3 = new mapping(p3);
    m4 = new mapping(p4);
    m5 = new mapping(p5);

    b0 = mapping::helper_create_block(MAP0_SIZE);
    b1 = mapping::helper_create_block(MAP1_SIZE);
    b2 = mapping::helper_create_block(MAP2_SIZE);
    b3 = mapping::helper_create_block(MAP3_SIZE);
    b4 = mapping::helper_create_block(MAP4_SIZE);
    b5 = mapping::helper_create_block(MAP5_SIZE);

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
                                                              MAP5_OFF -
                                                              MAP0_OFF);
    ASSERT_TRUE((*range.begin)->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range))->get_ptr() == p4);

    // CASE2
    // p0, p0 + (p5 - p0) + 1: [p0, p5] 
    //
    manager::range_mapping range2 = mgr->get_mappings_in_range(group, p0,
                                                               MAP5_OFF -
                                                               MAP0_OFF + 1);
    ASSERT_TRUE((*range2.begin)->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range2))->get_ptr() == p5);

    // CASE3
    // p0 + size0, p0 + (p5 - p0): [p1, p4] 
    //
    manager::range_mapping range3 = mgr->get_mappings_in_range(group, p0 + MAP0_SIZE,
                                                               (MAP5_OFF -
                                                                MAP0_OFF) -
                                                               MAP0_SIZE);
    ASSERT_TRUE((*range3.begin)->get_ptr() == p1);
    ASSERT_TRUE((*get_last_in_range(range3))->get_ptr() == p4);

    // CASE4
    // p0 + size0 - 1, p0 + (p5 - p0): [p0, p4] 
    //
    manager::range_mapping range4 = mgr->get_mappings_in_range(group, p0 + (MAP0_SIZE - 1),
                                                               (MAP5_OFF -
                                                                MAP0_OFF) -
                                                                MAP0_SIZE);
    ASSERT_TRUE((*range4.begin)->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range4))->get_ptr() == p4);

    // CASE5
    // p3, p3 + size3: [p3, p3] 
    //
    manager::range_mapping range5 = mgr->get_mappings_in_range(group, p3,
                                                               MAP3_SIZE);
    ASSERT_TRUE((*range5.begin)->get_ptr() == p3);
    ASSERT_TRUE((*get_last_in_range(range5))->get_ptr() == p3);


    // CASE6
    // p3 - 1, p3 + size3 - 1: [p2, p3] 
    //
    manager::range_mapping range6 = mgr->get_mappings_in_range(group, p3 - 1,
                                                               MAP3_SIZE);
    ASSERT_TRUE((*range6.begin)->get_ptr() == p2);
    ASSERT_TRUE((*get_last_in_range(range6))->get_ptr() == p3);

    // CASE7
    // p4 - 1, p4 + size4 - 1: [p4, p4] 
    //
    manager::range_mapping range7 = mgr->get_mappings_in_range(group, p4 - 1,
                                                               MAP4_SIZE);
    ASSERT_TRUE((*range7.begin)->get_ptr() == p4);
    ASSERT_TRUE((*get_last_in_range(range7))->get_ptr() == p4);

    range_fini();
}

static const int MAP_B0_OFF = 0x0900;
static const int MAP_B1_OFF = 0x1200;
static const int MAP_B2_OFF = 0x2000;
static const int MAP_B3_OFF = 0x2600;
static const int MAP_B4_OFF = 0x3000;
static const int MAP_B5_OFF = 0x3600;
static const int MAP_B6_OFF = 0x3950;
static const int MAP_B7_OFF = 0x5500;

static const size_t MAP_B0_SIZE = 0x0200;
static const size_t MAP_B1_SIZE = 0x0200;
static const size_t MAP_B2_SIZE = 0x0200;
static const size_t MAP_B3_SIZE = 0x0500;
static const size_t MAP_B4_SIZE = 0x0100;
static const size_t MAP_B5_SIZE = 0x0500;
static const size_t MAP_B6_SIZE = 0x0050;
static const size_t MAP_B7_SIZE = 0x0100;

static ptr         p_b0, p_b1, p_b2, p_b3, p_b4, p_b5, p_b6, p_b7;
static mapping_ptr m_b0, m_b1, m_b2, m_b3, m_b4, m_b5, m_b6, m_b7;
static block_ptr   b_b0, b_b1, b_b2, b_b3, b_b4, b_b5, b_b6, b_b7;

TEST_F(manager_mapping_test, insert_mappings)
{
    range_init();

    // We use the same base allocation
    p_b0 = p0 + (MAP_B0_OFF - MAP0_OFF);
    p_b1 = p0 + (MAP_B1_OFF - MAP0_OFF);
    p_b2 = p0 + (MAP_B2_OFF - MAP0_OFF);
    p_b3 = p0 + (MAP_B3_OFF - MAP0_OFF);
    p_b4 = p0 + (MAP_B4_OFF - MAP0_OFF);
    p_b5 = p0 + (MAP_B5_OFF - MAP0_OFF);
    p_b6 = p0 + (MAP_B5_OFF - MAP0_OFF);
    p_b7 = p0 + (MAP_B5_OFF - MAP0_OFF);

    m_b0 = new mapping(p_b0);
    m_b1 = new mapping(p_b1);
    m_b2 = new mapping(p_b2);
    m_b3 = new mapping(p_b3);
    m_b4 = new mapping(p_b4);
    m_b5 = new mapping(p_b5);
    m_b6 = new mapping(p_b6);
    m_b7 = new mapping(p_b7);

    b_b0 = mapping::helper_create_block(MAP_B0_SIZE);
    b_b1 = mapping::helper_create_block(MAP_B1_SIZE);
    b_b2 = mapping::helper_create_block(MAP_B2_SIZE);
    b_b3 = mapping::helper_create_block(MAP_B3_SIZE);
    b_b4 = mapping::helper_create_block(MAP_B4_SIZE);
    b_b5 = mapping::helper_create_block(MAP_B5_SIZE);
    b_b6 = mapping::helper_create_block(MAP_B6_SIZE);
    b_b7 = mapping::helper_create_block(MAP_B7_SIZE);

    gmacError_t err;
    err = m_b0->append(b_b0);
    ASSERT_TRUE(err == gmacSuccess);
    err = m_b1->append(b_b1);
    ASSERT_TRUE(err == gmacSuccess);
    err = m_b2->append(b_b2);
    ASSERT_TRUE(err == gmacSuccess);
    err = m_b3->append(b_b3);
    ASSERT_TRUE(err == gmacSuccess);
    err = m_b4->append(b_b4);
    ASSERT_TRUE(err == gmacSuccess);
    err = m_b5->append(b_b5);
    ASSERT_TRUE(err == gmacSuccess);
    err = m_b6->append(b_b6);
    ASSERT_TRUE(err == gmacSuccess);
    err = m_b7->append(b_b7);
    ASSERT_TRUE(err == gmacSuccess);

    delete m_b0;
    delete m_b1;
    delete m_b2;
    delete m_b3;
    delete m_b4;
    delete m_b5;
    delete m_b6;
    delete m_b7;

    range_fini();
}
