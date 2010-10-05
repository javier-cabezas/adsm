#include "Mode.h"
#include "Context.h"
#include "Accelerator.h"

namespace gmac { namespace cuda {

Mode::Mode(Process &proc, Accelerator &acc) :
    gmac::Mode(proc, acc)
{
#ifdef USE_MULTI_CONTEXT
    _cudaCtx = accelerator().createCUContext::current();
#endif
}

Mode::~Mode()
{
    ModuleVector::const_iterator m;
    switchIn();
#ifdef USE_MULTI_CONTEXT
    accelerator().destroyModules(modules);
    modules.clear();
#endif
    switchOut();
}

gmac::Context &Mode::getContext()
{
    gmac::Context *context = contextMap_.find(SELF());
    if(context != NULL) return *context;
    context = new Context(accelerator(), *this);
    switchIn();
    modules = accelerator().createModules();

    kernels_.clear();

    ModuleVector::const_iterator i;
#ifdef USE_MULTI_CONTEXT
    for(i = modules.begin(); i != modules.end(); i++) {
#else
    for(i = modules->begin(); i != modules->end(); i++) {
#endif
        (*i)->registerKernels(*this);
#ifdef USE_VM
        if((*i)->dirtyBitmap() != NULL) {
            _bitmapDevPtr = (*i)->dirtyBitmap()->devPtr();
            _bitmapShiftPageDevPtr = (*i)->dirtyBitmapShiftPage()->devPtr();
#ifdef BITMAP_BIT
            _bitmapShiftEntryDevPtr = (*i)->dirtyBitmapShiftEntry()->devPtr();
#endif
        }
#endif
    }
    switchOut();

    contextMap_.add(SELF(), context);
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

gmacError_t Mode::waitForBuffer(IOBuffer &buffer)
{
	switchIn();
    Context &ctx = dynamic_cast<Context &>(getContext());
	error_ = ctx.waitForBuffer(buffer);
	switchOut();
	return error_;
}

}}
