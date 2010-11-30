#include "Accelerator.h"
#include "Context.h"
#include "IOBuffer.h"
#include "Mode.h"

namespace __impl { namespace cuda {

Mode::Mode(core::Process &proc, Accelerator &acc) :
    core::Mode(proc, acc)
{
#ifdef USE_MULTI_CONTEXT
    cudaCtx_ = getAccelerator().createCUContext::current();
#endif
    switchIn();
    modules = getAccelerator().createModules();

    ModuleVector::const_iterator i;
#ifdef USE_MULTI_CONTEXT
    for(i = modules.begin(); i != modules.end(); i++) {
#else
    for(i = modules->begin(); i != modules->end(); i++) {
#endif
        (*i)->registerKernels(*this);
#ifdef USE_VM
        if((*i)->dirtyBitmap() != NULL) {
            bitmapAccPtr_ = (*i)->dirtyBitmap()->devPtr();
            bitmapShiftPageAccPtr_ = (*i)->dirtyBitmapShiftPage()->devPtr();
        }
#endif
    }

    void *addr = NULL;
    gmacError_t ret = hostAlloc(&addr, paramIOMemory);
    if(ret == gmacSuccess)
        ioMemory_ = new core::allocator::Buddy(addr, paramIOMemory);

    switchOut();
}

Mode::~Mode()
{
    // We need to ensure that contexts are destroyed before the Mode
    cleanUpContexts();

    ModuleVector::const_iterator m;
    switchIn();
#ifdef USE_MULTI_CONTEXT
    getAccelerator().destroyModules(modules);
    modules.clear();
#endif
    if(ioMemory_ != NULL) {
        hostFree(ioMemory_->addr());
        delete ioMemory_;
    }
    switchOut();
}

inline
core::IOBuffer *Mode::createIOBuffer(size_t size)
{
    if(ioMemory_ == NULL) return NULL;
    void *addr = ioMemory_->get(size);
    if(addr == NULL) return NULL;
    return new IOBuffer(addr, size);
}

inline
void Mode::destroyIOBuffer(core::IOBuffer *buffer)
{
    ASSERTION(ioMemory_ != NULL);
    ioMemory_->put(buffer->addr(), buffer->size());
    delete buffer;
}



void Mode::load()
{
#ifdef USE_MULTI_CONTEXT
    cudaCtx_ = getAccelerator().createCUContext::current();
#endif

    modules = getAccelerator().createModules();
    ModuleVector::const_iterator i;
#ifdef USE_MULTI_CONTEXT
    for(i = modules.begin(); i != modules.end(); i++) {
#else
    for(i = modules->begin(); i != modules->end(); i++) {
#endif
        (*i)->registerKernels(*this);
#ifdef USE_VM
        if((*i)->dirtyBitmap() != NULL) {
            bitmapAccPtr_ = (*i)->dirtyBitmap()->devPtr();
            bitmapShiftPageAccPtr_ = (*i)->dirtyBitmapShiftPage()->devPtr();
        }
#endif
    }

}

void Mode::reload()
{
#ifdef USE_MULTI_CONTEXT
    getAccelerator().destroyModules(modules);
    modules.clear();
#endif
    kernels_.clear();
    load();
}

core::Context &Mode::getContext()
{
	core::Context *context = contextMap_.find(util::GetThreadId());
    if(context != NULL) return *context;
    context = new cuda::Context(getAccelerator(), *this);
    CFATAL(context != NULL, "Error creating new context");
	contextMap_.add(util::GetThreadId(), context);
    return *context;
}

gmacError_t Mode::hostAlloc(void **addr, size_t size)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostAlloc(addr, size);
    switchOut();
    return ret;
}

gmacError_t Mode::hostFree(void *addr)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostFree(addr);
    switchOut();
    return ret;
}

void *Mode::hostMap(const void *addr)
{
    switchIn();
    void *ret = getAccelerator().hostMap(addr);
    switchOut();
    return ret;
}

const Variable *Mode::constant(gmacVariable_t key) const
{
    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    for(m = modules.begin(); m != modules.end(); m++) {
#else
    for(m = modules->begin(); m != modules->end(); m++) {
#endif
        const Variable *var = (*m)->constant(key);
        if(var != NULL) return var;
    }
    return NULL;
}

const Variable *Mode::variable(gmacVariable_t key) const
{
    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    for(m = modules.begin(); m != modules.end(); m++) {
#else
    for(m = modules->begin(); m != modules->end(); m++) {
#endif
        const Variable *var = (*m)->variable(key);
        if(var != NULL) return var;
    }
    return NULL;
}

const Texture *Mode::texture(gmacTexture_t key) const
{
    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    for(m = modules.begin(); m != modules.end(); m++) {
#else
    for(m = modules->begin(); m != modules->end(); m++) {
#endif
        const Texture *tex = (*m)->texture(key);
        if(tex != NULL) return tex;
    }
    return NULL;
}

CUstream Mode::eventStream()
{
    Context &ctx = dynamic_cast<Context &>(getContext());
    return ctx.eventStream();
}

gmacError_t Mode::waitForEvent(CUevent event)
{
	switchIn();
    Accelerator &acc = dynamic_cast<Accelerator &>(getAccelerator());

    CUresult ret;
    while ((ret = acc.queryCUevent(event)) == CUDA_ERROR_NOT_READY) {
        // TODO: add delay here
    }

	switchOut();

    return Accelerator::error(ret);
}

}}
