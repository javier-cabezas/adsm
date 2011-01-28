//si added// 
#ifndef TEST_GMAC_CORE_CONTEXT_H_
#define TEST_GMAC_CORE_CONTEXT_H_

#include "unit/init.h"
#include "gtest/gtest.h"



class ContextTest : public testing::Test {
protected:
	static const int Size_ = 4 * 1024 * 1024;

	static void TearDownTestCase() {
		FiniProcess();		
	}

	void SetUp() {
        InitProcess();
	}
};

#endif

