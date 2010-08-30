#include "Mode.h"
#include "Context.h"
#include "Accelerator.h"

namespace gmac { namespace gpu {

Mode::Mode(Accelerator *acc) :
    gmac::Mode(acc),
    acc(acc)
{
#ifdef USE_MULTI_CONTEXT
    __ctx = acc->createContext();
#endif

    switchIn();
    context = new Context(acc, this);
    switchOut();
    gmac::Mode::context = context;
}

gmacError_t Mode::hostAlloc(void **addr, size_t size)
{
    switchIn();
#if CUDART_VERSION >= 2020
    CUresult ret = cuMemHostAlloc(addr, size, CU_MEMHOSTALLOC_PORTABLE);
#else
    CUresult ret = cuMemAllocHost(addr, size);
#endif
    switchOut();
    return Accelerator::error(ret);
}

gmacError_t Mode::hostFree(void *addr)
{
    switchIn();
    CUresult r = cuMemFreeHost(addr);
    switchOut();
    return Accelerator::error(r);
}

const Variable *Mode::constant(gmacVariable_t key) const
{
    return context->constant(key);
}

const Variable *Mode::variable(gmacVariable_t key) const
{
    return context->variable(key);
}

const Texture *Mode::texture(gmacTexture_t key) const
{
    return context->texture(key);
}


Stream Mode::eventStream() const {
    return context->eventStream();
}

}}
