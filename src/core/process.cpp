#include "core/process.h"

#include "memory/Handler.h"

namespace __impl { namespace core {

process::process()
{
    memory::Handler::setProcess(*this);
}

process::~process()
{
}

}}
