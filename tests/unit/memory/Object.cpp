#include "gtest/gtest.h"

#include "core/IOBuffer.h"
#include "core/Mode.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"
#include "core/hpe/Thread.h"
#include "memory/Manager.h"
#include "memory/Object.h"

using gmac::core::hpe::Thread;

using __impl::core::Mode;
using __impl::memory::object;

extern void OpenCL(gmac::core::hpe::Process &);
extern void CUDA(gmac::core::hpe::Process &);

class ObjectTest : public testing::Test {
protected:
    static gmac::core::hpe::Process *Process_;
    static gmac::memory::manager *Manager_;
        static const size_t Size_;

        static void SetUpTestCase();
        static void TearDownTestCase();
};


gmac::core::hpe::Process *ObjectTest::Process_ = NULL;
gmac::memory::manager *ObjectTest::Manager_ = NULL;
const size_t ObjectTest::Size_ = 4 * 1024 * 1024;

void ObjectTest::SetUpTestCase()
{
    Process_ = new gmac::core::hpe::Process();
    ASSERT_TRUE(Process_ != NULL);
#if defined(USE_CUDA)
    CUDA(*Process_);
#endif
#if defined(USE_OPENCL)
    OpenCL(*Process_);
#endif
    Manager_ = new gmac::memory::manager(*Process_);
}

void ObjectTest::TearDownTestCase()
{
    ASSERT_TRUE(Manager_ != NULL);
    Manager_->destroy();
    Manager_ = NULL;

    ASSERT_TRUE(Process_ != NULL);
    Process_->destroy();
    Process_ = NULL;
}

TEST_F(ObjectTest, Creation)
{
    ASSERT_TRUE(Process_ != NULL);
    Mode &mode = Thread::getCurrentVirtualDevice();
    __impl::memory::ObjectMap &map = mode.getAddressSpace();
    __impl::memory::protocol &proto = map.get_protocol();
    object *object = proto.create_object(Process_->getResourceManager(), Size_, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(object != NULL);
    ASSERT_TRUE(object->addr() != NULL);
    ASSERT_TRUE(object->end() != NULL);
    ASSERT_EQ(Size_, size_t(object->end() - object->addr()));
    ASSERT_EQ(Size_, object->size());

    map.removeObject(*object);
    object->decRef();
}

TEST_F(ObjectTest, Blocks)
{
    ASSERT_TRUE(Process_ != NULL);
    Mode &mode = Thread::getCurrentVirtualDevice();
    __impl::memory::ObjectMap &map = mode.getAddressSpace();
    object *object = map.get_protocol().create_object(Process_->getResourceManager(), Size_, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(object != NULL);
    host_ptr start = object->addr();
    ASSERT_TRUE(start != NULL);
    host_ptr end = object->end();
    ASSERT_TRUE(end != NULL);
    size_t get_block_size = object->get_block_size();
    ASSERT_GT(get_block_size, 0U);

    for(size_t offset = 0; offset < object->size(); offset += get_block_size) {
        EXPECT_EQ(0, object->get_block_base(offset));
        EXPECT_EQ(get_block_size, object->get_block_end(offset));
    }

    map.removeObject(*object);
    object->decRef();
}

TEST_F(ObjectTest, Coherence)
{
    ASSERT_TRUE(Process_ != NULL);
    Mode &mode = Thread::getCurrentVirtualDevice();
    __impl::memory::ObjectMap &map = mode.getAddressSpace();
    object *object = map.get_protocol().create_object(Process_->getResourceManager(), Size_, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(object != NULL);
    object->add_owner(mode);
    map.addObject(*object);

    host_ptr ptr = object->addr();
    for(size_t s = 0; s < object->size(); s++) {
       ptr[s] = (s & 0xff);
    }
    ASSERT_EQ(gmacSuccess, object->release());
    ASSERT_EQ(gmacSuccess, object->to_device());

    GmacProtection prot = GMAC_PROT_READWRITE;
    ASSERT_EQ(gmacSuccess, object->acquire(prot));
    mode.memset(object->acceleratorAddr(mode, object->addr()), 0, Size_);

    for(size_t s = 0; s < object->size(); s++) {
        EXPECT_EQ(ptr[s], 0);
    }

    map.removeObject(*object);
    object->decRef();
}

TEST_F(ObjectTest, IOBuffer)
{
    ASSERT_TRUE(Process_ != NULL);
    Mode &mode = Thread::getCurrentVirtualDevice();
    __impl::memory::ObjectMap &map = mode.getAddressSpace();
    object *object = map.get_protocol().create_object(Process_->getResourceManager(), Size_, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(object != NULL);
    object->add_owner(mode);
    map.addObject(*object);

    __impl::core::IOBuffer &buffer = mode.createIOBuffer(Size_, GMAC_PROT_READWRITE);

    host_ptr ptr = buffer.addr();
    for(size_t s = 0; s < buffer.size(); s++) {
        ptr[s] = (s & 0xff);
    }

    ASSERT_EQ(gmacSuccess, object->copyFromBuffer(buffer, Size_));

    ptr = buffer.addr();
    memset(ptr, 0, Size_);

    ASSERT_EQ(gmacSuccess, object->copyToBuffer(buffer, Size_));
    ASSERT_EQ(gmacSuccess, buffer.wait());

    ptr = buffer.addr();
    int error = 0;
    for(size_t s = 0; s < buffer.size(); s++) {
        //EXPECT_EQ(ptr[s], (s & 0xff));
        error += (ptr[s] - (s & 0xff));
    }
    EXPECT_EQ(error, 0);

    mode.destroyIOBuffer(buffer);

    map.removeObject(*object);
    object->decRef();
}
