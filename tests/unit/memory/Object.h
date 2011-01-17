
#ifndef TEST_GMAC_MEMORY_OBJECT_H_
#define TEST_GMAC_MEMORY_OBJECT_H_

#include "unit/init.h"
#include "core/Process.h"
#include "core/Mode.h"
#include "memory/Object.h"
#include "memory/Block.h"
#include "gtest/gtest.h"

using gmac::memory::Protocol;
using gmac::core::Process;
using gmac::memory::Object;
using gmac::memory::Block;
using gmac::core::Mode;

class ObjectTest : public testing::Test{

public:
	static gmac::memory::Protocol *Protocol_;
	static gmac::memory::Object *Object_;

	const static int Size_ = 4 * 1024 * 1024;

	static void SetUpTestCase()
	{
		InitProcess();
		if(Protocol_ != NULL) return;
		Protocol_ = &Process::getInstance().protocol();
		ASSERT_TRUE(Protocol_ != NULL);
		Object_= Protocol_->createObject(Size_*sizeof(int),NULL,GMAC_PROT_NONE,0);
		ASSERT_TRUE( Object_ != NULL);

	}


	static void TearDownTestCase() {
		ASSERT_TRUE(Object_ != NULL);
		Protocol_->deleteObject(*Object_);
		FiniProcess();
		Protocol_ = NULL;
	}

};

#endif
