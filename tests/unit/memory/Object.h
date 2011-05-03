#ifndef TEST_GMAC_MEMORY_OBJECT_H_
#define TEST_GMAC_MEMORY_OBJECT_H_

#include "gtest/gtest.h"

#include "core/hpe/Process.h"

class ObjectTest : public testing::Test {
protected:
    static gmac::core::hpe::Process *Process_;
	static const size_t Size_;

	static void SetUpTestCase();
	static void TearDownTestCase();
};

#endif
