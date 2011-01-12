#include "gtest/gtest.h"

#include "init.h"

int main(int argc, char *argv[])
{
    InitGmac();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
