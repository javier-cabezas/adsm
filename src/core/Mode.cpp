#include "memory/Manager.h"
#include "memory/Object.h"
#include "memory/Protocol.h"

#include "trace/Tracer.h"

#include "Accelerator.h"
#include "IOBuffer.h"
#include "Kernel.h"
#include "Mode.h"
#include "Process.h"

namespace __impl { namespace core {

util::Private<Mode> Mode::key;

unsigned Mode::next = 0;

Mode::Mode(Process &proc, Accelerator &acc) :
    id_(++next),
    proc_(proc),
    acc_(&acc),
#if defined(_MSC_VER)
#	pragma warning( disable : 4355 )
#endif
    map_("ModeMemoryMap", *this),
    releasedObjects_(false)
#ifdef USE_SUBBLOCK_TRACKING
    , hostBitmap_(*this)
#else
#ifdef USE_VM
    , acceleratorBitmap_(*this)
#endif
#endif
{
    TRACE(LOCAL,"Creating Execution Mode %p", this);
    protocol_ = memory::ProtocolInit(0);
}

Mode::~Mode()
{    
    delete protocol_;
    if(this == key.get()) key.set(NULL);
    contextMap_.clean();
    acc_->unregisterMode(*this); 
    Process::getInstance().removeMode(*this);
    TRACE(LOCAL,"Destroying Execution Mode %p", this);
}

void Mode::finiThread()
{
    Mode *mode = key.get();
    if(mode == NULL) return;
    mode->release();
}

void Mode::kernel(gmacKernel_t k, Kernel &kernel)
{
    TRACE(LOCAL,"CTX: %p Registering kernel %s: %p", this, kernel.name(), k);
    KernelMap::iterator i;
    i = kernels_.find(k);
    ASSERTION(i == kernels_.end());
    kernels_[k] = &kernel;
}

Mode &Mode::getCurrent()
{
    Mode *mode = Mode::key.get();
    if(mode == NULL) {
        Process &proc = Process::getInstance();
        mode = proc.createMode();
    }
    ASSERTION(mode != NULL);
    return *mode;
}

void Mode::attach()
{
    Mode *mode = Mode::key.get();
    if(mode == this) return;
    if(mode != NULL) mode->release();
    key.set(this);
}

void Mode::detach()
{
    Mode *mode = Mode::key.get();
    if(mode != NULL) mode->release();
    key.set(NULL);
}

gmacError_t Mode::malloc(accptr_t &addr, size_t size, unsigned align)
{
    switchIn();
    error_ = acc_->malloc(addr, size, align);
    switchOut();
    return error_;
}

gmacError_t Mode::free(accptr_t addr)
{
    switchIn();
    error_ = acc_->free(addr);
    switchOut();
    return error_;
}

gmacError_t Mode::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    TRACE(LOCAL,"Copy %p to accelerator %p ("FMT_SIZE" bytes)", host, acc.get(), size);
    uint64_t start, end;
    trace::TimeMark(start);

    switchIn();
    error_ = getContext().copyToAccelerator(acc, host, size);
    switchOut();

    return error_;
}

gmacError_t Mode::copyToHost(hostptr_t host, const accptr_t acc, size_t size)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", acc.get(), host, size);
    uint64_t start, end;
    trace::TimeMark(start);

    switchIn();
    error_ = getContext().copyToHost(host, acc, size);
    switchOut();

    return error_;
}

gmacError_t Mode::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    switchIn();
    error_ = getContext().copyAccelerator(dst, src, size);
    switchOut();
    return error_;
}

gmacError_t Mode::memset(accptr_t addr, int c, size_t size)
{
    switchIn();
    error_ = getContext().memset(addr, c, size);
    switchOut();
    return error_;
}

// Nobody can enter GMAC until this has finished. No locks are needed
gmacError_t Mode::moveTo(Accelerator &acc)
{
    TRACE(LOCAL,"Moving mode from acc %d to %d", acc_->id(), acc.id());
    switchIn();

    if (acc_ == &acc) {
        switchOut();
        return gmacSuccess;
    }
    gmacError_t ret = gmacSuccess;
    size_t free;
    size_t total;
    size_t needed = map_.memorySize();
    acc_->memInfo(free, total);

    if (needed > free) {
        switchOut();
        return gmacErrorInsufficientAcceleratorMemory;
    }

    TRACE(LOCAL,"Releasing object memory in accelerator");
    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    map_.forEach(&memory::Object::unmapFromAccelerator);

    TRACE(LOCAL,"Cleaning contexts");
    contextMap_.clean();

    TRACE(LOCAL,"Registering mode in new accelerator");
    acc_->unregisterMode(*this);
    acc_ = &acc;
    acc_->registerMode(*this);
    
    TRACE(LOCAL,"Reallocating objects");
    //map_.reallocObjects(*this);
    map_.forEach(&memory::Object::mapToAccelerator);

    TRACE(LOCAL,"Reloading mode");
    reload();

    switchOut();

    return ret;
}

}}
