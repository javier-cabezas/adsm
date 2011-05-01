#include "gtest/gtest.h"

#include "core/Mode.h"

class IOBufferTest : public testing::Test {
protected:
    static const size_t Size_ = 4 * 1024 * 1024;

    static void ToAccelerator(__impl::core::Mode &mode);
    static void ToHost(__impl::core::Mode &mode);
};

