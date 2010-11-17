#include "memory/Manager.h"
#include "trace/Tracer.h"

#include "Accelerator.h"
#include "Context.h"

namespace gmac { namespace core {

Context::Context(Accelerator &acc, unsigned id) :
    util::RWLock("Context"),
    acc_(acc),
    id_(id)
{
	trace::StartThread(THREAD_T(id_), "GPU");
}

Context::~Context()
{ 
	trace::EndThread(THREAD_T(id_));
}

Context &Context::operator =(const Context &)
{
    FATAL("Assigment of contexts is not supported");
    return *this;
}

void
Context::init()
{
}

gmacError_t Context::copyToAccelerator(void *dev, const void *host, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = acc_.copyToAccelerator(dev, host, size);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyToHost(void *host, const void *dev, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = acc_.copyToHost(host, dev, size);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyAccelerator(void *dst, const void *src, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = acc_.copyAccelerator(dst, src, size);
    trace::ExitCurrentFunction();
    return ret;
}

}}
