#ifndef TEST_GMAC_CORE_CONTEXT_H_
#define TEST_GMAC_CORE_CONTEXT_H_

#include "gtest/gtest.h"
#include "core/hpe/Process.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Context.h"

class ContextTest : public testing::Test {
protected:
    static gmac::core::hpe::Process *Process_;

	static const int Size_ = 4 * 1024 * 1024;

    static void Memory(__impl::core::hpe::Mode &, gmac::core::hpe::Context &);

};

#endif

