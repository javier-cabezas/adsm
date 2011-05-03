#include "unit/memory/Object.h"

#include "core/Mode.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"
#include "memory/Object.h"

using __impl::core::Mode;
using __impl::memory::Object;

extern void OpenCL(gmac::core::hpe::Process &);
extern void CUDA(gmac::core::hpe::Process &);

gmac::core::hpe::Process *ObjectTest::Process_ = NULL;
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
}

void ObjectTest::TearDownTestCase()
{
    ASSERT_TRUE(Process_ != NULL);
    Process_->destroy();
    Process_ = NULL;
}

TEST_F(ObjectTest, Creation)
{
    ASSERT_TRUE(Process_ != NULL);
    Mode &mode = Process_->getCurrentMode();
    Object *object = mode.protocol().createObject(mode, Size_, NULL, GMAC_PROT_READ, 0);
    ASSERT_TRUE(object != NULL);
    ASSERT_TRUE(object->addr() != NULL);
    ASSERT_TRUE(object->end() != NULL);
    ASSERT_EQ(Size_, size_t(object->end() - object->addr()));
    ASSERT_EQ(Size_, object->size());

    mode.removeObject(*object);
    object->release();
}

