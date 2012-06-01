#include "gtest/gtest.h"

class hal_exec_test :
    public testing::Test {
protected:
    static void SetUpTestCase();
    static void TearDownTestCase();
};

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
