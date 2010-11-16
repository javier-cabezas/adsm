#include "memory/Manager.h"
#include "trace/Thread.h"
#include "trace/Function.h"

#include "Accelerator.h"
#include "Context.h"

namespace gmac { namespace core {

Context::Context(Accelerator &acc, unsigned id) :
    util::RWLock("Context"),
    acc_(acc),
    id_(id)
{
    gmac::trace::Thread::start((THREAD_T)id);
}

Context::~Context()
{ 
    gmac::trace::Thread::end((THREAD_T)id_);
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
    trace::Function::start("Context", "copyToAccelerator");
    gmacError_t ret = acc_.copyToAccelerator(dev, host, size);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::copyToHost(void *host, const void *dev, size_t size)
{
    trace::Function::start("Context", "copyToHost");
    gmacError_t ret = acc_.copyToHost(host, dev, size);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::copyAccelerator(void *dst, const void *src, size_t size)
{
    trace::Function::start("Context", "copyAccelerator");
    gmacError_t ret = acc_.copyAccelerator(dst, src, size);
    trace::Function::end("Context");
    return ret;
}

}}
