#include "api/opencl/hpe/Mode.h"
#include "api/opencl/hpe/ModeFactory.h"

namespace __impl { namespace opencl { namespace hpe {

Mode *
ModeFactory::create(core::hpe::Process &proc, core::hpe::AddressSpace &aSpace) const
{
    return new gmac::opencl::hpe::Mode(proc, aSpace);
}

}}}
