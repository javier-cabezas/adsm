#include "memory/Manager.h"
#include "trace/Thread.h"
#include "trace/Function.h"

#include "Accelerator.h"
#include "Context.h"

namespace gmac {

Context::Context(Accelerator &acc, unsigned id) :
    util::RWLock("Context"),
    acc_(acc),
    id_(id)
{
    gmac::trace::Thread::start(id);
}

Context::~Context()
{ 
    gmac::trace::Thread::end(id_);
}


void
Context::init()
{
}


gmacError_t Context::copyToDevice(void *dev, const void *host, size_t size)
{
    trace::Function::start("Context", "copyToDevice");
    gmacError_t ret = acc_.copyToDevice(dev, host, size);
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

gmacError_t Context::copyDevice(void *dst, const void *src, size_t size)
{
    trace::Function::start("Context", "copyDevice");
    gmacError_t ret = acc_.copyDevice(dst, src, size);
    trace::Function::end("Context");
    return ret;
}

}
