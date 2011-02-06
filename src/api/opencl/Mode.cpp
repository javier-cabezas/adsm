#include "Accelerator.h"
#include "Context.h"
#include "IOBuffer.h"
#include "Mode.h"

namespace __impl { namespace opencl {

Mode::Mode(core::Process &proc, Accelerator &acc) :
    gmac::core::Mode(proc, acc)
{
    hostptr_t addr = NULL;
    gmacError_t ret = hostAlloc(addr, util::params::ParamIOMemory);
    if(ret == gmacSuccess)
        ioMemory_ = new core::allocator::Buddy(addr, util::params::ParamIOMemory);
    else ioMemory_ = NULL;
}

Mode::~Mode()
{
    // We need to ensure that contexts are destroyed before the Mode
    cleanUpContexts();

    if(ioMemory_ != NULL) {
        hostFree(ioMemory_->addr());
        delete ioMemory_;
    }
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


void Mode::reload()
{
}

core::Context &Mode::getContext()
{
	core::Context *context = contextMap_.find(util::GetThreadId());
    if(context != NULL) return *context;
    context = new opencl::Context(getAccelerator(), *this);
    CFATAL(context != NULL, "Error creating new context");
	contextMap_.add(util::GetThreadId(), context);
    return *context;
}

Context &Mode::getCLContext()
{
    return dynamic_cast<Context &>(getContext());
}

gmacError_t Mode::hostAlloc(hostptr_t &addr, size_t size)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostAlloc(addr, size);
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

accptr_t Mode::hostMap(const hostptr_t addr, size_t size)
{
    switchIn();
    accptr_t ret = getAccelerator().hostMap(addr, size);
    switchOut();
    return ret;
}

accptr_t Mode::hostMapAddr(const hostptr_t addr)
{
    switchIn();
    accptr_t ret = getAccelerator().hostMapAddr(addr);
    switchOut();
    return ret;
}

cl_command_queue Mode::eventStream()
{
    Context &ctx = getCLContext();
    return ctx.eventStream();
}

gmacError_t Mode::waitForEvent(cl_event event)
{
	switchIn();
    Accelerator &acc = dynamic_cast<Accelerator &>(getAccelerator());

    gmacError_t ret = acc.syncCLevent(event);
    switchOut();
    return ret;

    // TODO: try to implement wait as a polling loop -- AMD OpenCL blocks execution
#if 0
    cl_int ret;
    while ((ret = acc.queryCLevent(event)) != CL_COMPLETE) {
        // TODO: add delay here
    }

	switchOut();

    return Accelerator::error(ret);
#endif
}

}}
