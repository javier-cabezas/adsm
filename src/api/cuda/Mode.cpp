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
    ioBuffer = new IOBuffer(paramBufferPageLockedSize * paramPageSize);
    gmac::Mode::context = context;
    modules = ModuleDescriptor::createModules(*this);
    switchOut();
}

Mode::~Mode()
{
    ModuleVector::const_iterator m;
    switchIn();
    for(m = modules.begin(); m != modules.end(); m++) {
        delete (*m);
    }
    delete context;
    delete ioBuffer;
    switchOut();
    modules.clear();
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

void *Mode::hostAddress(void *addr)
{
    switchIn();
    CUdeviceptr device;
    CUresult ret = cuMemHostGetDevicePointer(&device, addr, 0);
    if(ret != CUDA_SUCCESS) device = NULL;
    switchOut();
    return (void *)device;
}

const Variable *Mode::constant(gmacVariable_t key) const
{
    ModuleVector::const_iterator m;
    for(m = modules.begin(); m != modules.end(); m++) {
        const Variable *var = (*m)->constant(key);
        if(var != NULL) return var;
    }
    return NULL;
}

const Variable *Mode::variable(gmacVariable_t key) const
{
    ModuleVector::const_iterator m;
    for(m = modules.begin(); m != modules.end(); m++) {
        const Variable *var = (*m)->variable(key);
        if(var != NULL) return var;
    }
    return NULL;
}

const Texture *Mode::texture(gmacTexture_t key) const
{
    ModuleVector::const_iterator m;
    for(m = modules.begin(); m != modules.end(); m++) {
        const Texture *tex = (*m)->texture(key);
        if(tex != NULL) return tex;
    }
    return NULL;
}

Stream Mode::eventStream() const {
    return context->eventStream();
}

}}
