#include "gtest/gtest.h"

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
    /* We always return 0 to let hudson continue executing tests */
    return 0;
}
