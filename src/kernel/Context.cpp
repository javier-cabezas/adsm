#include "Context.h"

#include <memory/Manager.h>
#include <config/paraver.h>

namespace gmac {

Context::Context(Accelerator *acc, unsigned id) :
    util::RWLock("Context"),
    acc(acc),
    id(id)
{
    addThreadTid(0x10000000 + id);
    pushState(Idle, 0x10000000 + id);
}

Context::~Context() { }


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
