#include "Mode.h"
#include "Context.h"
#include "Accelerator.h"

namespace __impl { namespace cuda {

Mode::Mode(core::Process &proc, Accelerator &acc) :
    __impl::core::Mode(proc, acc)
{
#ifdef USE_MULTI_CONTEXT
    cudaCtx_ = accelerator().createCUContext::current();
#endif
    switchIn();
    modules = accelerator().createModules();

    ModuleVector::const_iterator i;
#ifdef USE_MULTI_CONTEXT
    for(i = modules.begin(); i != modules.end(); i++) {
#else
    for(i = modules->begin(); i != modules->end(); i++) {
#endif
        (*i)->registerKernels(*this);
#ifdef USE_VM
        if((*i)->dirtyBitmap() != NULL) {
            bitmapDevPtr_ = (*i)->dirtyBitmap()->devPtr();
            bitmapShiftPageDevPtr_ = (*i)->dirtyBitmapShiftPage()->devPtr();
#ifdef BITMAP_BIT
            bitmapShiftEntryDevPtr_ = (*i)->dirtyBitmapShiftEntry()->devPtr();
#endif
        }
#endif
    }

    void *addr = NULL;
    gmacError_t ret = hostAlloc(&addr, paramIOMemory);
    if(ret == gmacSuccess)
        ioMemory_ = new __impl::core::allocator::Buddy(addr, paramIOMemory);

    switchOut();
}

Mode::~Mode()
{
    // We need to ensure that contexts are destroyed before the Mode
    cleanUpContexts();

    ModuleVector::const_iterator m;
    switchIn();
#ifdef USE_MULTI_CONTEXT
    accelerator().destroyModules(modules);
    modules.clear();
#endif
    if(ioMemory_ != NULL) {
        hostFree(ioMemory_->addr());
        delete ioMemory_;
    }
    switchOut();
}

void Mode::load()
{
#ifdef USE_MULTI_CONTEXT
    cudaCtx_ = accelerator().createCUContext::current();
#endif

    modules = accelerator().createModules();
    ModuleVector::const_iterator i;
#ifdef USE_MULTI_CONTEXT
    for(i = modules.begin(); i != modules.end(); i++) {
#else
    for(i = modules->begin(); i != modules->end(); i++) {
#endif
        (*i)->registerKernels(*this);
#ifdef USE_VM
        if((*i)->dirtyBitmap() != NULL) {
            bitmapDevPtr_ = (*i)->dirtyBitmap()->devPtr();
            bitmapShiftPageDevPtr_ = (*i)->dirtyBitmapShiftPage()->devPtr();
#ifdef BITMAP_BIT
            bitmapShiftEntryDevPtr_ = (*i)->dirtyBitmapShiftEntry()->devPtr();
#endif
        }
#endif
    }

}

void Mode::reload()
{
#ifdef USE_MULTI_CONTEXT
    accelerator().destroyModules(modules);
    modules.clear();
#endif
    kernels_.clear();
    load();
}

core::Context &Mode::getContext()
{
	core::Context *context = contextMap_.find(util::GetThreadId());
    if(context != NULL) return *context;
    context = new __impl::cuda::Context(accelerator(), *this);
    CFATAL(context != NULL, "Error creating new context");
	contextMap_.add(util::GetThreadId(), context);
    return *context;
}

gmacError_t Mode::hostAlloc(void **addr, size_t size)
{
    switchIn();
    gmacError_t ret = accelerator().hostAlloc(addr, size);
    switchOut();
    return ret;
}

gmacError_t Mode::hostFree(void *addr)
{
    switchIn();
    gmacError_t ret = accelerator().hostFree(addr);
    switchOut();
    return ret;
}

void *Mode::hostMap(void *addr)
{
    switchIn();
    void *ret = accelerator().hostMap(addr);
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

gmacError_t Mode::waitForBuffer(core::IOBuffer &buffer)
{
	switchIn();
    Context &ctx = dynamic_cast<Context &>(getContext());
	error_ = ctx.waitForBuffer(buffer);
	switchOut();
	return error_;
}

}}
