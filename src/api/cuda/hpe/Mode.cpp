#include "api/cuda/hpe/Accelerator.h"
#include "api/cuda/hpe/Context.h"
#include "api/cuda/hpe/Mode.h"


#include "api/cuda/IOBuffer.h"

namespace __impl { namespace cuda { namespace hpe {

Mode::Mode(core::hpe::Process &proc, Accelerator &acc) :
    gmac::core::hpe::Mode(proc, acc)
#ifdef USE_VM
    , bitmap_(*this)
#endif
{
#ifdef USE_MULTI_CONTEXT
    cudaCtx_ = getAccelerator().createCUcontext();
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
    }

    hostptr_t addr = NULL;
    gmacError_t ret = hostAlloc(addr, util::params::ParamIOMemory);
    if(ret == gmacSuccess)
        ioMemory_ = new core::allocator::Buddy(addr, util::params::ParamIOMemory);

    switchOut();
}

Mode::~Mode()
{
    switchIn();

    // We need to ensure that contexts are destroyed before the Mode
    cleanUpContexts();

    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    getAccelerator().destroyModules(modules);
    modules.clear();
#endif
    if(ioMemory_ != NULL) {
        hostFree(ioMemory_->addr());
        delete ioMemory_;
        ioMemory_ = NULL;
    }

    switchOut();
}

inline
core::IOBuffer &Mode::createIOBuffer(size_t size)
{
    IOBuffer *ret;
    void *addr;
    if(ioMemory_ == NULL || (addr = ioMemory_->get(size)) == NULL) {
        addr = ::malloc(size);
        ret = new IOBuffer(addr, size, false);
    } else {
        ret = new IOBuffer(addr, size, true);
    }
    return *ret;
}

inline
void Mode::destroyIOBuffer(core::IOBuffer &buffer)
{
    ASSERTION(ioMemory_ != NULL);
    if (buffer.async()) {
        ioMemory_->put(buffer.addr(), buffer.size());
    } else {
        ::free(buffer.addr());
    }
    delete &buffer;
}

void Mode::load()
{
#ifdef USE_MULTI_CONTEXT
    cudaCtx_ = getAccelerator().createCUcontext();
#endif

    modules = getAccelerator().createModules();
    ModuleVector::const_iterator i;
#ifdef USE_MULTI_CONTEXT
    for(i = modules.begin(); i != modules.end(); i++) {
#else
    for(i = modules->begin(); i != modules->end(); i++) {
#endif
        (*i)->registerKernels(*this);
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

core::hpe::Context &Mode::getContext()
{
	core::hpe::Context *context = contextMap_.find(util::GetThreadId());
    if(context != NULL) return *context;
    context = new cuda::hpe::Context(getAccelerator(), *this);
    CFATAL(context != NULL, "Error creating new context");
	contextMap_.add(util::GetThreadId(), context);
    return *context;
}

Context &Mode::getCUDAContext()
{
    return dynamic_cast<Context &>(getContext());
}

gmacError_t Mode::hostAlloc(hostptr_t &addr, size_t size)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostAlloc(&addr, size);
    switchOut();
    return ret;
}

gmacError_t Mode::hostFree(hostptr_t addr)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostFree(addr);
    switchOut();
    return ret;
}

gmacError_t
Mode::map(accptr_t &dst, hostptr_t src, size_t size, unsigned align)
{
    switchIn();

    accptr_t acc(0);
    bool hasMapping = acc_->getMapping(acc, src, size);
    if (hasMapping == true) {
        error_ = gmacSuccess;
        dst = acc;
        TRACE(LOCAL,"Mapping for address %p: %u:%p", src, dst.pasId_, dst.get());
    } else {
        error_ = acc_->map(dst, src, size, align);
        TRACE(LOCAL,"New Mapping for address %p: %u:%p", src, dst.pasId_, dst.get());
    }
#ifdef USE_MULTI_CONTEXT
    dst.pasId_ = id_;
#endif

    switchOut();
    return error_;
}

accptr_t Mode::hostMapAddr(const hostptr_t addr)
{
    switchIn();
    accptr_t ret = getAccelerator().hostMap(addr);
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

const Variable *Mode::constantByName(std::string name) const
{
    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    for(m = modules.begin(); m != modules.end(); m++) {
#else
    for(m = modules->begin(); m != modules->end(); m++) {
#endif
        const Variable *var = (*m)->constantByName(name);
        if(var != NULL) return var;
    }
    return NULL;
}

const Variable *Mode::variableByName(std::string name) const
{
    ModuleVector::const_iterator m;
#ifdef USE_MULTI_CONTEXT
    for(m = modules.begin(); m != modules.end(); m++) {
#else
    for(m = modules->begin(); m != modules->end(); m++) {
#endif
        const Variable *var = (*m)->variableByName(name);
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
    Context &ctx = getCUDAContext();
    return ctx.eventStream();
}

gmacError_t Mode::waitForEvent(CUevent event, bool fromCUDA)
{
    // Backend methods do not need to switch in/out
    if (!fromCUDA) switchIn();
    Accelerator &acc = dynamic_cast<Accelerator &>(getAccelerator());

    CUresult ret;
    while ((ret = acc.queryCUevent(event)) == CUDA_ERROR_NOT_READY) {
        // TODO: add delay here
    }

    if (!fromCUDA) switchOut();
    return Accelerator::error(ret);
}

}}}
