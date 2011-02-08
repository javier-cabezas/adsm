#include <map>
#include <string>
#include <vector>

#include "common.h"

TestSuite Tests("GMAC");

static void
LaunchTests()
{
    TestSuite suite("GMAC");
    ReadConf(suite);

    suite.launch();
    suite.report();
}

int main(int argc, char *argv[])
{
    LaunchTests();
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
