#include "Context.h"

#include <memory/Manager.h>
#include <trace/Thread.h>

namespace gmac {

Context::Context(Accelerator *acc, unsigned id) :
    util::RWLock("Context"),
    acc(acc),
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
    return acc->copyToDevice(dev, host, size);
}

gmacError_t Context::copyToHost(void *host, const void *dev, size_t size)
{
    return acc->copyToHost(host, dev, size);
}

gmacError_t Context::copyDevice(void *dst, const void *src, size_t size)
{
    return acc->copyDevice(dst, src, size);
}


}
