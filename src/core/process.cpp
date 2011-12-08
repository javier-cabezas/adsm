#include "core/process.h"

#include "memory/handler.h"

namespace __impl { namespace core {

process::process()
{
    memory::handler::setProcess(*this);
}

process::~process()
{
}

}}
