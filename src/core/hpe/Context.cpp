#include "memory/Manager.h"
#include "trace/Tracer.h"

#include "core/Accelerator.h"
#include "core/hpe/Context.h"

namespace __impl { namespace core { namespace hpe {

Context::Context(Accelerator &acc, Mode &mode, unsigned id) :
    gmac::util::RWLock("Context"),
    acc_(acc),
    mode_(mode),
    id_(id)
{
}

Context::~Context()
{ 
}

void
Context::init()
{
}

gmacError_t Context::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = acc_.copyToAccelerator(acc, host, size, mode_);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyToHost(hostptr_t host, const accptr_t acc, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = acc_.copyToHost(host, acc, size, mode_);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = acc_.copyAccelerator(dst, src, size);
    trace::ExitCurrentFunction();
    return ret;
}

}}}
