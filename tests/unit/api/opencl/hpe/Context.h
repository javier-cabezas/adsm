#ifndef TEST_GMAC_CORE_CONTEXT_H_
#define TEST_GMAC_CORE_CONTEXT_H_

#include "gtest/gtest.h"
#include "core/hpe/Process.h"


class OpenCLContextTest : public testing::Test, ContextTest {
protected:
    static gmac::core::hpe::Process *Process_;

	static const int Size_ = 4 * 1024 * 1024;

	static void SetUpTestCase();
	static void TearDownTestCase();

};

#endif

