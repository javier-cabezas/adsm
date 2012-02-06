#include "unit2/dsm/manager.h"

#include "hal/types.h"

#include "util/misc.h"

#include "gtest/gtest.h"

void
manager_test::SetUpTestCase()
{
}

void manager_test::TearDownTestCase()
{
}

TEST_F(manager_test, mappings_in_range)
{
    static const unsigned MAP0_ADDR  = 0x0100;
    static const unsigned MAP0_SIZE  = 0x0100;
    static const unsigned MAP1_ADDR  = 0x1000;
    static const unsigned MAP1_SIZE  = 0x0200;
    static const unsigned MAP2_ADDR  = 0x2000;
    static const unsigned MAP2_SIZE  = 0x1000;
    static const unsigned MAP3_ADDR  = 0x3000;
    static const unsigned MAP3_SIZE  = 0x0900;
    static const unsigned MAP4_ADDR  = 0x4000;
    static const unsigned MAP4_SIZE  = 0x0700;
    static const unsigned MAP5_ADDR  = 0x5000;
    static const unsigned MAP5_SIZE  = 0x0800;

    manager_ptr mgr = new manager();

    __impl::hal::context_t *ctx1 = 0;// create_context();

    ptr p0 = ptr(ptr::backend_ptr(MAP0_ADDR), ctx1);
    ptr p1 = p0 + (MAP1_ADDR - MAP0_ADDR);
    ptr p2 = p0 + (MAP2_ADDR - MAP0_ADDR);
    ptr p3 = p0 + (MAP3_ADDR - MAP0_ADDR);
    ptr p4 = p0 + (MAP4_ADDR - MAP0_ADDR);
    ptr p5 = p0 + (MAP5_ADDR - MAP0_ADDR);

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

    gmacError_t err;

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

    mgr->helper_insert(*ctx1, m0);
    mgr->helper_insert(*ctx1, m1);
    mgr->helper_insert(*ctx1, m2);
    mgr->helper_insert(*ctx1, m3);
    mgr->helper_insert(*ctx1, m4);
    mgr->helper_insert(*ctx1, m5);

    delete mgr;
}
