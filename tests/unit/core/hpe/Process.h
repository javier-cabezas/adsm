#ifndef TEST_GMAC_CORE_PROCESS_H_
#define TEST_GMAC_CORE_PROCESS_H_


#include "core/Process.h"
#include "core/Mode.h"


#include "gtest/gtest.h"



class ProcessTest: public testing::Test {

public:

	static void SetUpTestCase() {
        InitProcess();
    }

	static void TearDownTestCase() {
		FiniProcess();
	}



};

#endif
