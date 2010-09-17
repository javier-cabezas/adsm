#include "Context.h"

#include <memory/Manager.h>
#include <trace/Thread.h>
#include <trace/Function.h>

namespace gmac {

Context::Context(Accelerator *acc, unsigned id) :
    util::RWLock("Context"),
    _acc(acc),
    id(id)
{
    gmac::trace::Thread::start(id);
}

Context::~Context()
{ 
    gmac::trace::Thread::end(id);
}


gmacError_t Context::copyToDevice(void *dev, const void *host, size_t size)
{
    trace::Function::start("Context", "copyToDevice");
    gmacError_t ret = _acc->copyToDevice(dev, host, size);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::copyToHost(void *host, const void *dev, size_t size)
{
    trace::Function::start("Context", "copyToHost");
    gmacError_t ret = _acc->copyToHost(host, dev, size);
    trace::Function::end("Context");
    return ret;
}

gmacError_t Context::copyDevice(void *dst, const void *src, size_t size)
{
    trace::Function::start("Context", "copyDevice");
    gmacError_t ret = _acc->copyDevice(dst, src, size);
    trace::Function::end("Context");
    return ret;
}

}
