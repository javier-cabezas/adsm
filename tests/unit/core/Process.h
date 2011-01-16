#ifndef TEST_GMAC_CORE_PROCESS_H_
#define TEST_GMAC_CORE_PROCESS_H_


#include "core/Process.h"
#include "core/Mode.h"


#include "gtest/gtest.h"



class ProcessTest: public testing::Test{

public:
	static gmac::core::Mode *Mode_;


	static void SetUpTestCase();

	static void TearDownTestCase() {
		Mode_->detach();
		FiniProcess();
		Mode_ = NULL;
	}



};

#endif