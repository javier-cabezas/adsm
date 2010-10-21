#ifdef DEBUG
#include "Contract.h"

#include <iostream>
#include <cassert>

namespace gmac { namespace test {

void Contract::Preamble(const char *file, const int line)
{
    std::cerr << "[Breach of Contract @ " << file << ":" << line << " ";
}


void Contract::Ensures(const char *file, const int line,
        const char *clause, bool b)
{
    if(b == true) return;
    Preamble(file, line);
    std::cerr << "Ensure " << clause << " not met" << std::endl;
    assert(0);
}

void Contract::Requires(const char *file, const int line,
        const char *clause, bool b)
{
    if(b == true) return;
    Preamble(file, line);
    std::cerr << "Requires " << clause << " not met" << std::endl;
    assert(0);
}

void Contract::Assert(const char *file, const int line,
        const char *clause, bool b)
{
    if(b == true) return;
    Preamble(file, line);
    std::cerr << "Assert " << clause << " not met" << std::endl;
    assert(0);
}


} }
#endif
