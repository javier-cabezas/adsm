#include "gtest/gtest.h"

#include "unit2/dsm/manager.h"

#include "util/misc.h"

#include "hal/error.h"

static I_HAL::virt::aspace *as0;

static manager *mgr = NULL;

static const int MAP0_OFF =  100;
static const int MAP1_OFF = 1000;
static const int MAP2_OFF = 2000;
static const int MAP3_OFF = 3000;
static const int MAP4_OFF = 4000;
static const int MAP5_OFF = 5000;

static const size_t MAP0_SIZE =  100;
static const size_t MAP1_SIZE =  200;
static const size_t MAP2_SIZE = 1000;
static const size_t MAP3_SIZE =  900;
static const size_t MAP4_SIZE =  700;
static const size_t MAP5_SIZE =  800;

void
manager_mapping_test::SetUpTestCase()
{
    mgr = new manager();
    as0 = I_HAL::virt::aspace::create();
}

void manager_mapping_test::TearDownTestCase()
{
    I_HAL::virt::aspace::destroy(*as0);
}

static I_HAL::virt::object_view *view0;

static ptr         p0, p1, p2, p3, p4, p5;
static mapping_ptr m0, m1, m2, m3, m4, m5;
static block_ptr   b0, b1, b2, b3, b4, b5;

static void range_init()
{
    view0 = new I_HAL::virt::object_view(*as0, 0);
    ptr ptrBase(*view0, 0);

    p0 = ptrBase + MAP0_OFF;
    p1 = p0 + (MAP1_OFF - MAP0_OFF);
    p2 = p0 + (MAP2_OFF - MAP0_OFF);
    p3 = p0 + (MAP3_OFF - MAP0_OFF);
    p4 = p0 + (MAP4_OFF - MAP0_OFF);
    p5 = p0 + (MAP5_OFF - MAP0_OFF);

    m0 = new mapping(p0, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m1 = new mapping(p1, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m2 = new mapping(p2, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m3 = new mapping(p3, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m4 = new mapping(p4, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m5 = new mapping(p5, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);

    b0 = mapping::helper_create_block(MAP0_SIZE);
    b1 = mapping::helper_create_block(MAP1_SIZE);
    b2 = mapping::helper_create_block(MAP2_SIZE);
    b3 = mapping::helper_create_block(MAP3_SIZE);
    b4 = mapping::helper_create_block(MAP4_SIZE);
    b5 = mapping::helper_create_block(MAP5_SIZE);

    I_DSM::error err;
    bool berr;

    err = m0->append(b0);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m1->append(b1);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m2->append(b2);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m3->append(b3);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m4->append(b4);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m5->append(b5);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);

    berr = mgr->helper_insert(*as0, m0);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as0, m1);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as0, m2);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as0, m3);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as0, m4);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as0, m5);
    ASSERT_TRUE(berr);
}

static void range_fini()
{
    bool berr;

    berr = mgr->helper_delete_mappings(*as0);
    ASSERT_TRUE(berr);

    delete view0;
}

template <typename T>
T
get_last_in_range(range<T> &r)
{
    for (typename range<T>::iterator it =  r.begin();
                                     it != r.end();
                                   ++it) {
        typename range<T>::iterator it2 = it;
        ++it2;
        if (it2 == r.end()) return it;
    }
    return r.end();
}


TEST_F(manager_mapping_test, mappings_in_range)
{
    range_init();

    manager::map_mapping_group &group = mgr->get_aspace_mappings(*as0);

    // CASE1
    // p0, p0 + (p5 - p0): [p0, p4]
    //
    manager::range_mapping range = mgr->get_mappings_in_range<false>(group, p0,
                                                                     MAP5_OFF -
                                                                     MAP0_OFF);

    ASSERT_TRUE((*range.begin())->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range))->get_ptr() == p4);

    // CASE2
    // p0, p0 + (p5 - p0) + 1: [p0, p5] 
    //
    manager::range_mapping range2 = mgr->get_mappings_in_range<false>(group, p0,
                                                                      MAP5_OFF -
                                                                      MAP0_OFF + 1);
    ASSERT_TRUE((*range2.begin())->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range2))->get_ptr() == p5);

    // CASE3
    // p0 + size0, p0 + (p5 - p0): [p1, p4] 
    //
    manager::range_mapping range3 = mgr->get_mappings_in_range<false>(group, p0 + MAP0_SIZE,
                                                                      (MAP5_OFF -
                                                                       MAP0_OFF) -
                                                                      MAP0_SIZE);
    ASSERT_TRUE((*range3.begin())->get_ptr() == p1);
    ASSERT_TRUE((*get_last_in_range(range3))->get_ptr() == p4);

    // CASE4
    // p0 + size0 - 1, p0 + (p5 - p0): [p0, p4] 
    //
    manager::range_mapping range4 = mgr->get_mappings_in_range<false>(group, p0 + (MAP0_SIZE - 1),
                                                                      (MAP5_OFF -
                                                                       MAP0_OFF) -
                                                                      MAP0_SIZE);
    ASSERT_TRUE((*range4.begin())->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range4))->get_ptr() == p4);

    // CASE5
    // p3, p3 + size3: [p3, p3] 
    //
    manager::range_mapping range5 = mgr->get_mappings_in_range<false>(group, p3,
                                                                      MAP3_SIZE);
    ASSERT_TRUE((*range5.begin())->get_ptr() == p3);
    ASSERT_TRUE((*get_last_in_range(range5))->get_ptr() == p3);


    // CASE6
    // p3 - 1, p3 + size3 - 1: [p2, p3] 
    //
    manager::range_mapping range6 = mgr->get_mappings_in_range<false>(group, p3 - 1,
                                                                      MAP3_SIZE);
    ASSERT_TRUE((*range6.begin())->get_ptr() == p2);
    ASSERT_TRUE((*get_last_in_range(range6))->get_ptr() == p3);

    // CASE7
    // p4 - 1, p4 + size4 - 1: [p4, p4] 
    //
    manager::range_mapping range7 = mgr->get_mappings_in_range<false>(group, p4 - 1,
                                                                      MAP4_SIZE);
    ASSERT_TRUE((*range7.begin())->get_ptr() == p4);
    ASSERT_TRUE((*get_last_in_range(range7))->get_ptr() == p4);

    range_fini();
}

TEST_F(manager_mapping_test, mappings_in_range_boundaries)
{
    range_init();

    manager::map_mapping_group &group = mgr->get_aspace_mappings(*as0);

    // CASE1
    // p0, p0 + (p5 - p0): [p0, p5]
    //
    manager::range_mapping range = mgr->get_mappings_in_range<true>(group, p0,
                                                                    MAP5_OFF -
                                                                    MAP0_OFF);
    ASSERT_TRUE((*range.begin())->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range))->get_ptr() == p5);

    // CASE2
    // p0, p0 + (p5 - p0) + 1: [p0, p5]
    //
    manager::range_mapping range2 = mgr->get_mappings_in_range<true>(group, p0,
                                                                     MAP5_OFF -
                                                                     MAP0_OFF + 1);
    ASSERT_TRUE((*range2.begin())->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range2))->get_ptr() == p5);

    // CASE3
    // p0 + size0, p0 + (p5 - p0): [p1, p5] 
    //
    manager::range_mapping range3 = mgr->get_mappings_in_range<true>(group, p0 + MAP0_SIZE,
                                                                     (MAP5_OFF -
                                                                      MAP0_OFF) -
                                                                      MAP0_SIZE);
    ASSERT_TRUE((*range3.begin())->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range3))->get_ptr() == p5);

    // CASE4
    // p0 + size0 - 1, p0 + (p5 - p0): [p0, p4] 
    //
    manager::range_mapping range4 = mgr->get_mappings_in_range<true>(group, p0 + (MAP0_SIZE - 1),
                                                                     (MAP5_OFF -
                                                                      MAP0_OFF) -
                                                                     MAP0_SIZE);
    ASSERT_TRUE((*range4.begin())->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range4))->get_ptr() == p4);

    // CASE5
    // p3, p3 + size3: [p3, p3] 
    //
    manager::range_mapping range5 = mgr->get_mappings_in_range<true>(group, p3,
                                                                     MAP3_SIZE);
    ASSERT_TRUE((*range5.begin())->get_ptr() == p2);
    ASSERT_TRUE((*get_last_in_range(range5))->get_ptr() == p3);


    // CASE6
    // p3 - 1, p3 + size3 - 1: [p2, p3] 
    //
    manager::range_mapping range6 = mgr->get_mappings_in_range<true>(group, p3 - 1,
                                                                     MAP3_SIZE);
    ASSERT_TRUE((*range6.begin())->get_ptr() == p2);
    ASSERT_TRUE((*get_last_in_range(range6))->get_ptr() == p3);

    // CASE7
    // p4 - 1, p4 + size4 - 1: [p4, p4] 
    //
    manager::range_mapping range7 = mgr->get_mappings_in_range<true>(group, p4 - 1,
                                                                     MAP4_SIZE);
    ASSERT_TRUE((*range7.begin())->get_ptr() == p4);
    ASSERT_TRUE((*get_last_in_range(range7))->get_ptr() == p4);

    range_fini();
}

static const int MAP_B0_OFF =   50;
static const int MAP_B1_OFF =  900;
static const int MAP_B2_OFF = 1200;
static const int MAP_B3_OFF = 2000;
static const int MAP_B4_OFF = 2600;
static const int MAP_B5_OFF = 3000;
static const int MAP_B6_OFF = 3600;
static const int MAP_B7_OFF = 3950;
static const int MAP_B8_OFF = 5500;

static const size_t MAP_B0_SIZE = 100;
static const size_t MAP_B1_SIZE = 200;
static const size_t MAP_B2_SIZE = 200;
static const size_t MAP_B3_SIZE = 200;
static const size_t MAP_B4_SIZE = 500;
static const size_t MAP_B5_SIZE = 100;
static const size_t MAP_B6_SIZE = 500;
static const size_t MAP_B7_SIZE =  50;
static const size_t MAP_B8_SIZE = 100;

static ptr         p_b0, p_b1, p_b2, p_b3, p_b4, p_b5, p_b6, p_b7, p_b8;
static mapping_ptr m_b0, m_b1, m_b2, m_b3, m_b4, m_b5, m_b6, m_b7, m_b8;

TEST_F(manager_mapping_test, insert_blocks)
{
    block_ptr b_b0, b_b1, b_b2, b_b3, b_b4, b_b5, b_b6, b_b7, b_b8;

    range_init();

    // We use the same base allocation
    p_b0 = p0 + (MAP_B0_OFF - MAP0_OFF);
    p_b1 = p0 + (MAP_B1_OFF - MAP0_OFF);
    p_b2 = p0 + (MAP_B2_OFF - MAP0_OFF);
    p_b3 = p0 + (MAP_B3_OFF - MAP0_OFF);
    p_b4 = p0 + (MAP_B4_OFF - MAP0_OFF);
    p_b5 = p0 + (MAP_B5_OFF - MAP0_OFF);
    p_b6 = p0 + (MAP_B6_OFF - MAP0_OFF);
    p_b7 = p0 + (MAP_B7_OFF - MAP0_OFF);
    p_b8 = p0 + (MAP_B8_OFF - MAP0_OFF);

    m_b0 = new mapping(p_b0, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b1 = new mapping(p_b1, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b2 = new mapping(p_b2, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b3 = new mapping(p_b3, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b4 = new mapping(p_b4, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b5 = new mapping(p_b5, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b6 = new mapping(p_b6, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b7 = new mapping(p_b7, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b8 = new mapping(p_b8, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);

    b_b0 = mapping::helper_create_block(MAP_B0_SIZE);
    b_b1 = mapping::helper_create_block(MAP_B1_SIZE);
    b_b2 = mapping::helper_create_block(MAP_B2_SIZE);
    b_b3 = mapping::helper_create_block(MAP_B3_SIZE);
    b_b4 = mapping::helper_create_block(MAP_B4_SIZE);
    b_b5 = mapping::helper_create_block(MAP_B5_SIZE);
    b_b6 = mapping::helper_create_block(MAP_B6_SIZE);
    b_b7 = mapping::helper_create_block(MAP_B7_SIZE);
    b_b8 = mapping::helper_create_block(MAP_B8_SIZE);

    I_DSM::error err;
    err = m_b0->append(b_b0);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b1->append(b_b1);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b2->append(b_b2);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b3->append(b_b3);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b4->append(b_b4);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b5->append(b_b5);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b6->append(b_b6);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b7->append(b_b7);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b8->append(b_b8);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);

    bool berr;
    berr = mgr->helper_insert(*as0, m_b0);
    ASSERT_FALSE(berr);
    delete m_b0;
    berr = mgr->helper_insert(*as0, m_b1);
    ASSERT_FALSE(berr);
    delete m_b1;
    berr = mgr->helper_insert(*as0, m_b2);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as0, m_b3);
    ASSERT_FALSE(berr);
    delete m_b3;
    berr = mgr->helper_insert(*as0, m_b4);
    ASSERT_FALSE(berr);
    delete m_b4;
    berr = mgr->helper_insert(*as0, m_b5);
    ASSERT_FALSE(berr);
    delete m_b5;
    berr = mgr->helper_insert(*as0, m_b6);
    ASSERT_FALSE(berr);
    delete m_b6;
    berr = mgr->helper_insert(*as0, m_b7);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as0, m_b8);
    ASSERT_FALSE(berr);
    delete m_b8;

    range_fini();
}

TEST_F(manager_mapping_test, insert_mappings)
{
    block_ptr b_b0, b_b1, b_b2, b_b3, b_b4, b_b5, b_b6, b_b7, b_b8;

    range_init();

    // We use the same base allocation
    p_b0 = p0 + (MAP_B0_OFF - MAP0_OFF);
    p_b1 = p0 + (MAP_B1_OFF - MAP0_OFF);
    p_b2 = p0 + (MAP_B2_OFF - MAP0_OFF);
    p_b3 = p0 + (MAP_B3_OFF - MAP0_OFF);
    p_b4 = p0 + (MAP_B4_OFF - MAP0_OFF);
    p_b5 = p0 + (MAP_B5_OFF - MAP0_OFF);
    p_b6 = p0 + (MAP_B6_OFF - MAP0_OFF);
    p_b7 = p0 + (MAP_B7_OFF - MAP0_OFF);
    p_b8 = p0 + (MAP_B8_OFF - MAP0_OFF);

    m_b0 = new mapping(p_b0, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b1 = new mapping(p_b1, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b2 = new mapping(p_b2, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b3 = new mapping(p_b3, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b4 = new mapping(p_b4, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b5 = new mapping(p_b5, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b6 = new mapping(p_b6, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b7 = new mapping(p_b7, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);
    m_b8 = new mapping(p_b8, GMAC_PROT_READWRITE, I_DSM::mapping_flags::MAP_DEFAULT);

    b_b0 = mapping::helper_create_block(MAP_B0_SIZE);
    b_b1 = mapping::helper_create_block(MAP_B1_SIZE);
    b_b2 = mapping::helper_create_block(MAP_B2_SIZE);
    b_b3 = mapping::helper_create_block(MAP_B3_SIZE);
    b_b4 = mapping::helper_create_block(MAP_B4_SIZE);
    b_b5 = mapping::helper_create_block(MAP_B5_SIZE);
    b_b6 = mapping::helper_create_block(MAP_B6_SIZE);
    b_b7 = mapping::helper_create_block(MAP_B7_SIZE);
    b_b8 = mapping::helper_create_block(MAP_B8_SIZE);

    I_DSM::error err;
    err = m_b0->append(b_b0);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b1->append(b_b1);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b2->append(b_b2);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b3->append(b_b3);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b4->append(b_b4);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b5->append(b_b5);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b6->append(b_b6);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b7->append(b_b7);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    err = m_b8->append(b_b8);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);

    bool berr;
    berr = mgr->helper_insert(*as0, m_b0);
    ASSERT_FALSE(berr);
    delete m_b0;
    berr = mgr->helper_insert(*as0, m_b1);
    ASSERT_FALSE(berr);
    delete m_b1;
    berr = mgr->helper_insert(*as0, m_b2);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as0, m_b3);
    ASSERT_FALSE(berr);
    delete m_b3;
    berr = mgr->helper_insert(*as0, m_b4);
    ASSERT_FALSE(berr);
    delete m_b4;
    berr = mgr->helper_insert(*as0, m_b5);
    ASSERT_FALSE(berr);
    delete m_b5;
    berr = mgr->helper_insert(*as0, m_b6);
    ASSERT_FALSE(berr);
    delete m_b6;
    berr = mgr->helper_insert(*as0, m_b7);
    ASSERT_TRUE(berr);
    berr = mgr->helper_insert(*as0, m_b8);
    ASSERT_FALSE(berr);
    delete m_b8;

    range_fini();
}



TEST_F(manager_mapping_test, merge_mappings)
{
    range_init();
    
    manager::map_mapping_group &group = mgr->get_aspace_mappings(*as0);

    manager::range_mapping range = mgr->get_mappings_in_range<true>(group, p0,
                                                                    MAP5_OFF -
                                                                    MAP0_OFF);

    ASSERT_TRUE(range.is_empty() == false);
    ASSERT_TRUE((*range.begin())->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range))->get_ptr() == p5);

    I_DSM::mapping_ptr merged = mgr->merge_mappings(range);
    ASSERT_TRUE(merged->get_bounds().get_size() == (MAP5_OFF - MAP0_OFF + MAP5_SIZE));
    ASSERT_TRUE(merged->get_nblocks() == 6 + 4);

    delete merged;

    range_fini();

    range_init();

    manager::range_mapping range2 = mgr->get_mappings_in_range<false>(group, p0,
                                                                      MAP5_OFF -
                                                                      MAP0_OFF);

    ASSERT_TRUE(range2.is_empty() == false);
    ASSERT_TRUE((*range2.begin())->get_ptr() == p0);
    ASSERT_TRUE((*get_last_in_range(range2))->get_ptr() == p4);

    I_DSM::mapping_ptr merged2 = mgr->merge_mappings(range2);
    ASSERT_TRUE(merged2->get_bounds().get_size() == (MAP4_OFF - MAP0_OFF + MAP4_SIZE));
    ASSERT_TRUE(merged2->get_nblocks() == 5 + 3);

    delete merged2;

    range_fini();
}

static I_HAL::virt::aspace *as1;

static I_HAL::virt::object_view *view1;
static I_HAL::virt::object_view *view2;
static I_HAL::virt::object_view *view3;
static I_HAL::virt::object_view *view4;

static ptr as0_p0,  as0_p1,  as0_p2;
static ptr as0_p0b, as0_p1b, as0_p2b;
static ptr as1_p0,  as1_p1,  as1_p2;
static ptr as1_p0b, as1_p1b, as1_p2b;

static const size_t LINK0_SIZE = 0x1000;
static const size_t LINK1_SIZE = 2 * 0x1000;
static const size_t LINK2_SIZE = 0x1000;

static const size_t LINK1_0_OFF = 2 * 0x1000;
static const size_t LINK1_1_OFF = 4 * 0x1000;
static const size_t LINK1_2_OFF = 7 * 0x1000;

static const size_t LINK2_0_OFF = 4 * 0x1000;
static const size_t LINK2_1_OFF = 6 * 0x1000;
static const size_t LINK2_2_OFF = 9 * 0x1000;

static void link_init()
{
    as1 = I_HAL::virt::aspace::create();

    I_HAL::virt::object_view *view1 = new I_HAL::virt::object_view(*as0, 0);
    ptr ptrBase1(*view1, 0);

    as0_p0  = ptrBase1 + LINK1_0_OFF;
    as0_p1  = ptrBase1 + LINK1_1_OFF;
    as0_p2  = ptrBase1 + LINK1_2_OFF;

    I_HAL::virt::object_view *view2 = new I_HAL::virt::object_view(*as0, 0x1000000);
    ptr ptrBase2(*view2, 0);

    as0_p0b = ptrBase2 + LINK1_0_OFF;
    as0_p1b = ptrBase2 + LINK1_1_OFF;
    as0_p2b = ptrBase2 + LINK1_2_OFF;

    // We use the same base allocation
    I_HAL::virt::object_view *view3 = new I_HAL::virt::object_view(*as1, 0);
    ptr ptrBase3(*view3, 0);

    as1_p0  = ptrBase3 + LINK2_0_OFF;
    as1_p1  = ptrBase3 + LINK2_1_OFF;
    as1_p2  = ptrBase3 + LINK2_2_OFF;

    I_HAL::virt::object_view *view4 = new I_HAL::virt::object_view(*as1, 0x1000000);
    ptr ptrBase4(*view4, 0);

    as1_p0b = ptrBase4 + LINK2_0_OFF;
    as1_p1b = ptrBase4 + LINK2_1_OFF;
    as1_p2b = ptrBase4 + LINK2_2_OFF;
}

static void link_fini()
{
    bool berr;
    berr = mgr->helper_delete_mappings(*as0);
    ASSERT_TRUE(berr);
    berr = mgr->helper_delete_mappings(*as1);
    ASSERT_TRUE(berr);

    delete view1;
    delete view2;
    delete view3;
    delete view4;

    I_HAL::virt::aspace::destroy(*as1);
}

TEST_F(manager_mapping_test, link)
{
    link_init();

    I_DSM::error err = mgr->link(as0_p0, as0_p0b, LINK0_SIZE, GMAC_PROT_READ, GMAC_PROT_READ);
    ASSERT_FALSE(err == I_DSM::error::DSM_SUCCESS);
    ASSERT_FALSE(mgr->helper_get_mapping(as0_p0));
    ASSERT_FALSE(mgr->helper_get_mapping(as0_p0b));

    err = mgr->link(as0_p0, as1_p0, LINK0_SIZE, GMAC_PROT_READ, GMAC_PROT_READ);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    ASSERT_TRUE(mgr->helper_get_mapping(as0_p0));
    ASSERT_TRUE(mgr->helper_get_mapping(as0_p0)->get_ptr() == as0_p0);
    ASSERT_TRUE(mgr->helper_get_mapping(as1_p0));
    ASSERT_TRUE(mgr->helper_get_mapping(as1_p0)->get_ptr() == as1_p0);
    ASSERT_TRUE(mgr->helper_get_mappings(as0_p0.get_view().get_vaspace(), as0_p0.get_view()).size() == 1);
    ASSERT_TRUE(mgr->helper_get_mappings(as1_p0.get_view().get_vaspace(), as1_p0.get_view()).size() == 1);

    link_fini();
}

TEST_F(manager_mapping_test, link2)
{
    link_init();

    I_DSM::error err = mgr->link(as0_p1, as1_p1, LINK0_SIZE, GMAC_PROT_READ, GMAC_PROT_READ);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    ASSERT_TRUE(mgr->helper_get_mapping(as0_p1));
    ASSERT_TRUE(mgr->helper_get_mapping(as0_p1)->get_ptr() == as0_p1);
    ASSERT_TRUE(mgr->helper_get_mapping(as1_p1));
    ASSERT_TRUE(mgr->helper_get_mapping(as1_p1)->get_ptr() == as1_p1);
    ASSERT_TRUE(mgr->helper_get_mappings(as0_p1.get_view().get_vaspace(), as0_p1.get_view()).size() == 1);
    ASSERT_TRUE(mgr->helper_get_mappings(as1_p1.get_view().get_vaspace(), as1_p1.get_view()).size() == 1);

    err = mgr->link(as0_p1, as1_p1, LINK2_2_OFF - LINK2_1_OFF +  LINK2_SIZE, GMAC_PROT_READ, GMAC_PROT_READ);
    ASSERT_TRUE(err == I_DSM::error::DSM_SUCCESS);
    ASSERT_TRUE(mgr->helper_get_mappings(as0_p1.get_view().get_vaspace(), as0_p1.get_view()).size() == 1);
    ASSERT_TRUE(mgr->helper_get_mappings(as1_p1.get_view().get_vaspace(), as1_p1.get_view()).size() == 1);

    link_fini();
}
