#include "unit2/dsm/mapping.h"

#include "hal/types.h"

#include "util/misc.h"

#include "gtest/gtest.h"

void
mapping_test::SetUpTestCase()
{
}

void mapping_test::TearDownTestCase()
{
}

typedef __impl::util::bounds<ptr> alloc;

TEST_F(mapping_test, prepend_block)
{
    static const unsigned BASE_ADDR = 0x100;
    static const unsigned OFFSET    = 0x400;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), (__impl::hal::context_t *)(0x1));

    mapping_ptr m0 = new mapping(p0 + OFFSET);

    block_ptr b0 = new block(0x100);
    block_ptr b1 = new block(0x300);

    gmacError_t err;

    ASSERT_TRUE(m0->get_bounds().get_size() == 0);
    err = m0->prepend(b0);
    ASSERT_TRUE(err == gmacSuccess);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());
    err = m0->prepend(b1);
    ASSERT_TRUE(err == gmacSuccess);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() + b1->get_size()));

    delete m0;

    delete b0;
    delete b1;
}

TEST_F(mapping_test, prepend_block2)
{
    static const unsigned BASE_ADDR = 0x100;
    static const unsigned OFFSET    = 0x400;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), (__impl::hal::context_t *)(0x1));

    mapping_ptr m0 = new mapping(p0 + OFFSET);

    block_ptr b0 = new block(0x100);
    block_ptr b1 = new block(0x400);

    gmacError_t err;

    ASSERT_TRUE(m0->get_bounds().get_size() == 0);
    err = m0->prepend(b0);
    ASSERT_TRUE(err == gmacSuccess);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());
    // The OFFSET 0x400 is not enough to fit the second block, TOTAL 0x500
    err = m0->prepend(b1);
    ASSERT_FALSE(err == gmacSuccess);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());

    delete m0;

    delete b0;
    delete b1;
}

TEST_F(mapping_test, append_block)
{
    static const unsigned BASE_ADDR = 0x100;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), (__impl::hal::context_t *)(0x1));

    mapping_ptr m0 = new mapping(p0);

    block_ptr b0 = new block(0x100);
    block_ptr b1 = new block(0x400);
    block_ptr b2 = new block(0x400);
    block_ptr b3 = new block(0x000);

    gmacError_t err;

    ASSERT_TRUE(m0->get_bounds().get_size() == 0);
    err = m0->append(b0);
    ASSERT_TRUE(err == gmacSuccess);
    ASSERT_TRUE(m0->get_bounds().get_size() == b0->get_size());
    err = m0->append(b1);
    ASSERT_TRUE(err == gmacSuccess);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size()));
    err = m0->append(b2);
    ASSERT_TRUE(err == gmacSuccess);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size() +
                                                b2->get_size()));
    err = m0->append(b3);
    // Blocks of size 0 are not allowed
    ASSERT_FALSE(err == gmacSuccess);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size() +
                                                b2->get_size()));

    delete m0;

    delete b0;
    delete b1;
    delete b2;
    delete b3;
}

TEST_F(mapping_test, append_mapping)
{
    static const unsigned BASE_ADDR  = 0x100;
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE + 0x100;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), (__impl::hal::context_t *)(0x1));
    ptr p1 = p0 + OFFSET;

    mapping_ptr m0 = new mapping(p0);
    mapping_ptr m1 = new mapping(p1);

    block_ptr b0 = new block(BLOCK_SIZE);
    block_ptr b1 = new block(BLOCK_SIZE);

    gmacError_t err;

    err = m0->append(b0);
    ASSERT_TRUE(err == gmacSuccess);
    err = m1->append(b1);
    ASSERT_TRUE(err == gmacSuccess);
    err = m0->append(m1);
    ASSERT_TRUE(err == gmacSuccess);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size() +
                                                (OFFSET - b0->get_size())));

    delete m0;
    delete m1;

    delete b0;
    delete b1;
}

TEST_F(mapping_test, append_mapping2)
{
    static const unsigned BASE_ADDR  = 0x100;
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE - 0x100;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), (__impl::hal::context_t *)(0x1));
    ptr p1 = p0 + OFFSET;

    mapping_ptr m0 = new mapping(p0);
    mapping_ptr m1 = new mapping(p1);

    block_ptr b0 = new block(BLOCK_SIZE);
    block_ptr b1 = new block(BLOCK_SIZE);

    gmacError_t err;

    err = m0->append(b0);
    ASSERT_TRUE(err == gmacSuccess);
    err = m1->append(b1);
    ASSERT_TRUE(err == gmacSuccess);
    err = m0->append(m1);
    ASSERT_FALSE(err == gmacSuccess);

    delete m0;
    delete m1;

    delete b0;
    delete b1;
}

TEST_F(mapping_test, append_mapping3)
{
    static const unsigned BASE_ADDR  = 0x100;
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), (__impl::hal::context_t *)(0x1));
    ptr p1 = p0 + OFFSET;

    mapping_ptr m0 = new mapping(p0);
    mapping_ptr m1 = new mapping(p1);

    block_ptr b0 = new block(BLOCK_SIZE);
    block_ptr b1 = new block(BLOCK_SIZE);

    gmacError_t err;

    err = m0->append(b0);
    ASSERT_TRUE(err == gmacSuccess);
    err = m1->append(b1);
    ASSERT_TRUE(err == gmacSuccess);
    err = m0->append(m1);
    ASSERT_TRUE(err == gmacSuccess);
    ASSERT_TRUE(m0->get_bounds().get_size() == (b0->get_size() +
                                                b1->get_size()));

    delete m0;
    delete m1;

    delete b0;
    delete b1;
}

TEST_F(mapping_test, append_mapping4)
{
    static const unsigned BASE_ADDR  = 0x100;
    static const unsigned BLOCK_SIZE = 0x300;
    static const unsigned OFFSET     = BLOCK_SIZE;

    ptr p0 = ptr(ptr::backend_ptr(BASE_ADDR), (__impl::hal::context_t *)(0x1));
    ptr p1 = ptr(ptr::backend_ptr(BASE_ADDR), (__impl::hal::context_t *)(0x2));

    mapping_ptr m0 = new mapping(p0);
    mapping_ptr m1 = new mapping(p1 + OFFSET);

    block_ptr b0 = new block(BLOCK_SIZE);
    block_ptr b1 = new block(BLOCK_SIZE);

    gmacError_t err;

    err = m0->append(b0);
    ASSERT_TRUE(err == gmacSuccess);
    err = m1->append(b1);
    ASSERT_TRUE(err == gmacSuccess);
    err = m0->append(m1);
    ASSERT_FALSE(err == gmacSuccess);

    delete m0;
    delete m1;

    delete b0;
    delete b1;
}

