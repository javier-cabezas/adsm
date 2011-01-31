#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#include "common.h"

static void
ReadVars(std::string fileName)
{
    std::ifstream input(fileName.c_str(), std::ios::in);
    assert(input.is_open());
    while (input.good()) {
        std::string line;
        std::getline(input, line);
        if (line.size() > 0) {
            if (line[0] == '#') continue;

            std::cout << line << std::endl;
            std::istringstream iss(line);

            std::string token;
            std::getline(iss, token, ':');
            std::cout << "Variable Name: " << token << std::endl;
            Variable v(token);

            while (std::getline(iss, token, ',')) {
                v += token;
                std::cout << " - " << token << std::endl;
            }
            Variables[v.getName()] = v;
        }
    }
}

static void
ReadTests(std::string fileName, TestSuite &suite)
{
    std::ifstream input(fileName.c_str(), std::ios::in);
    assert(input.is_open());
    while (input.good()) {
        std::string line;
        std::getline(input, line);
        if (line.size() > 0) {
            if (line[0] == '#') continue;

            std::cout << line << std::endl;
            std::istringstream iss(line);

            std::string token;
            std::getline(iss, token, ':');
            std::cout << "Name: " << token << std::endl;
            Test t(token);

            while (std::getline(iss, token, ',')) {
                if (Variables.find(token) == Variables.end()) {
                    std::cerr << "Variable '" << token << "' not found" << std::endl;
                    abort();
                }
                t += Variables[token];
                std::cout << " - " << token << std::endl;
            }
            suite += t;
        }
    }
    input.close();
}

void
ReadConf(TestSuite &suite)
{
    ReadVars("vars.spec");
    ReadTests("tests.spec", suite);
}
