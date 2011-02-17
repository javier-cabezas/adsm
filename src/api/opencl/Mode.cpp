#include "Accelerator.h"
#include "Context.h"
#include "IOBuffer.h"
#include "Mode.h"

namespace __impl { namespace opencl {

Mode::Mode(core::Process &proc, Accelerator &acc) :
    gmac::core::Mode(proc, acc),
	ioAddr_(NULL)
{
    gmacError_t ret = hostAlloc(ioAddr_, util::params::ParamIOMemory);
    if(ret == gmacSuccess)
        ioMemory_ = new core::allocator::Buddy(hostptr_t(0x1000), util::params::ParamIOMemory);
    else ioMemory_ = NULL;
}

Mode::~Mode()
{
    // We need to ensure that contexts are destroyed before the Mode
    cleanUpContexts();

    if(ioMemory_ != NULL && ioAddr_ != NULL) {
        hostFree(ioAddr_);
        delete ioMemory_;
    }
}

inline
core::IOBuffer &Mode::createIOBuffer(size_t size)
{
    IOBuffer *ret;
    void *addr = NULL;
    if(ioMemory_ == NULL || (addr = ioMemory_->get(size)) == NULL) {
        addr = ::malloc(size);
        ret = new IOBuffer(*this, NULL, hostptr_t(addr), size, false);
    } else {
        ret = new IOBuffer(*this, ioAddr_, hostptr_t(addr), size, true);
    }
    return *ret;
}

inline
void Mode::destroyIOBuffer(core::IOBuffer &buffer)
{
    ASSERTION(ioMemory_ != NULL);

    IOBuffer &ioBuffer = dynamic_cast<IOBuffer &>(buffer);
    if (ioBuffer.async()) {
        ioMemory_->put(hostptr_t(ioBuffer.offset() + 0x1000), ioBuffer.size());
    } else {
        ::free(buffer.addr());
    }
    delete &buffer;
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

gmacError_t Mode::hostAlloc(cl_mem &addr, size_t size)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostAlloc(addr, size);
    switchOut();
    return ret;
}

gmacError_t Mode::hostFree(cl_mem addr)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostFree(addr);
    switchOut();
    return ret;
}

hostptr_t Mode::hostMap(cl_mem addr, size_t offset, size_t size, cl_command_queue stream)
{
    switchIn();
    hostptr_t ret = getAccelerator().hostMap(addr, offset, size, stream);
    switchOut(); 
    return ret;
}

gmacError_t Mode::hostUnmap(hostptr_t ptr, cl_mem addr, size_t size, cl_command_queue stream)
{
    switchIn();
    gmacError_t ret = getAccelerator().hostUnmap(ptr, addr, size, stream);
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
