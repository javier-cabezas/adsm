#include "gtest/gtest.h"

#include "core/Mode.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"
#include "memory/Manager.h"

using __impl::core::Mode;
using __impl::core::Process;
using __impl::core::hpe::AddressSpace;
using __impl::memory::object;
using __impl::memory::Protocol;
using __impl::memory::ObjectMap;

extern void OpenCL(gmac::core::hpe::Process &);
extern void CUDA(gmac::core::hpe::Process &);

class ObjectMapTest : public testing::Test {
protected:
    static gmac::core::hpe::Process *Process_;
	static gmac::memory::manager *Manager_;
	static const size_t Size_;
	
	static void SetUpTestCase();
	static void TearDownTestCase();
};

gmac::core::hpe::Process *ObjectMapTest::Process_ = NULL;
gmac::memory::manager *ObjectMapTest::Manager_ = NULL;
const size_t ObjectMapTest::Size_ = 4 * 1024 * 1024;

void ObjectMapTest::SetUpTestCase()
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

void ObjectMapTest::TearDownTestCase()
{
    ASSERT_TRUE(Manager_ != NULL);
    Manager_->destroy();
    Manager_ = NULL;

    ASSERT_TRUE(Process_ != NULL);
    Process_->destroy();
    Process_ = NULL;
}

TEST_F(ObjectMapTest, Creation) {
	const char * name = "NameOfObjectMap";
	ObjectMap *map = new ObjectMap(name);
	ASSERT_TRUE(map != NULL);
	map->cleanUp();
}

TEST_F(ObjectMapTest, Coherence) {
	Mode *mode = Process_->createMode(0);
	ASSERT_TRUE(mode != NULL);

    ObjectMap &map = mode->getAddressSpace();
	ASSERT_TRUE(&map != NULL);	

	protocol &proto = map.get_protocol();
	ASSERT_TRUE(&proto != NULL);
	
    object *obj1 = proto.create_object(*mode, Size_, NULL, GMAC_PROT_READ, 0);
	object *obj2 = proto.create_object(*mode, Size_, NULL, GMAC_PROT_READWRITE, 0);

	host_ptr addr1 = obj1->addr();
	host_ptr addr2 = obj2->addr();

	ASSERT_FALSE(map.has_modified_objects());
	ASSERT_TRUE(map.size() == 0);
    ASSERT_TRUE(map.get_memory_size() == 0);
	ASSERT_EQ(gmacSuccess, map.releaseObjects());
	ASSERT_TRUE(map.released_objects());
	ASSERT_EQ(gmacSuccess, map.acquire_objects());	

	ASSERT_TRUE(map.addObject(*obj1));	
	ASSERT_TRUE(map.has_modified_objects());
	ASSERT_TRUE(map.size() == 1);
	ASSERT_TRUE(map.get_memory_size() == Size_);
	ASSERT_EQ(gmacSuccess, map.releaseObjects());
	ASSERT_TRUE(map.released_objects());
	ASSERT_EQ(gmacSuccess, map.acquire_objects());
	
	ASSERT_TRUE(map.hasObject(*obj1));
	ASSERT_TRUE(map.hasObject(*obj2) == 0);

	ASSERT_EQ(obj1, map.get_object(addr1, Size_));
	ASSERT_NE(obj1, map.get_object(addr2, Size_));

    ASSERT_TRUE(map.addObject(*obj2));
	ASSERT_TRUE(map.has_modified_objects());
	ASSERT_TRUE(map.size() == 2);
	ASSERT_TRUE(map.get_memory_size() == 2 * Size_);	
	ASSERT_EQ(gmacSuccess, map.releaseObjects());
	ASSERT_TRUE(map.released_objects());
    ASSERT_EQ(gmacSuccess, map.acquire_objects());

	ASSERT_TRUE(map.removeObject(*obj1));
	ASSERT_FALSE(map.has_modified_objects());
	ASSERT_TRUE(map.size() == 1);
	ASSERT_TRUE(map.get_memory_size() == Size_);	
	ASSERT_EQ(gmacSuccess, map.releaseObjects());
	ASSERT_TRUE(map.released_objects());
    ASSERT_EQ(gmacSuccess, map.acquire_objects());

    ASSERT_TRUE(map.removeObject(*obj2));
	ASSERT_FALSE(map.has_modified_objects());
	ASSERT_TRUE(map.size() == 0);
    ASSERT_TRUE(map.get_memory_size() == 0);	
	ASSERT_EQ(gmacSuccess, map.releaseObjects());
	ASSERT_TRUE(map.released_objects());
	ASSERT_EQ(gmacSuccess, map.acquire_objects());

    obj1->decRef();
	obj2->decRef();

	map.cleanUp();
}