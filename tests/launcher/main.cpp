#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "common.h"

TestSuite Tests("GMAC");

static void
LaunchTests(const char *varsPath, const char *testsPath)
{
    TestSuite suite("GMAC");
    ReadConf(suite, varsPath, testsPath);

    suite.launch();
    suite.report();
}

int main(int argc, char *argv[])
{
    if (argc == 1) {
        LaunchTests("vars.spec", "tests.spec");
    } else if (argc == 3) {
        LaunchTests(argv[1], argv[2]);
    } else {
        std::cerr << "Error: wrong number of parameters" << std::endl;
        std::cerr << " > launcher [ vars_file tests_file ]" << std::endl;
    }

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
