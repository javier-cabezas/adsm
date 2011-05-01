#include "core/Process.h"

#include "memory/Handler.h"

namespace __impl { namespace core {

Process::Process()
{
    memory::Handler::setProcess(*this);
}

Process::~Process()
{
    std::vector<library_t>::const_iterator i;
    for(i = handlers_.begin(); i != handlers_.end(); i++)
        RELEASE_LIBRARY(*i);
}

void Process::addHandler(library_t handler)
{
    handlers_.push_back(handler);
}

}}
