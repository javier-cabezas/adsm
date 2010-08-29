#include "Context.h"

#include <config.h>

#include <memory/Manager.h>
#include <gmac/init.h>

namespace gmac { namespace gpu {

Context::AddressMap Context::hostMem;
void * Context::FatBin;

Context::Context(Accelerator *acc) :
    gmac::Context(acc)
{
    _modules = ModuleDescriptor::createModules(*this);
}

Context::~Context()
{
    trace("Remove Accelerator context [%p]", this);
    ModuleVector::const_iterator m;
    for(m = _modules.begin(); m != _modules.end(); m++) {
        delete (*m);
    }
    _modules.clear();
}


}}
