#include "memory/Manager.h"
#include "memory/Object.h"
#include "memory/Protocol.h"

#include "Accelerator.h"
#include "IOBuffer.h"
#include "Kernel.h"
#include "Mode.h"
#include "Process.h"

namespace gmac { namespace core {

gmac::util::Private<Mode> Mode::key;

unsigned Mode::next = 0;

Mode::Mode(Process &proc, Accelerator &acc) :
    id_(++next),
    proc_(proc),
    acc_(&acc),
#if defined(_MSC_VER)
#	pragma warning( disable : 4355 )
#endif
    map_("ModeMemoryMap", *this),
    releasedObjects_(true)
#ifdef USE_VM
    , _bitmap(new memory::vm::Bitmap())
#endif
    , count_(0)
{
    TRACE(LOCAL,"Creating new memory map");
}

Mode::~Mode()
{
    count_--;
    if(count_ > 0)
        WARNING("Deleting in-use Execution Mode (%d)", count_);
    if(this == key.get()) key.set(NULL);
    contextMap_.clean();
}

void Mode::cleanUpContexts()
{
    contextMap_.clean();
}

void Mode::finiThread()
{
    Mode *mode = key.get();
    if(mode == NULL) return;
    memory::Manager &manager = memory::Manager::getInstance();
    mode->map_.forEach(manager.protocol(), &gmac::memory::Protocol::toHost);
    mode->map_.makeOrphans();

    memory::Manager::getInstance().removeMode(*mode);
    Process::getInstance().removeMode(mode);
}



void Mode::release()
{
#ifdef USE_VM
    delete _bitmap;
#endif
    acc_->unregisterMode(*this); 
}
void Mode::kernel(gmacKernel_t k, Kernel &kernel)
{
    TRACE(LOCAL,"CTX: %p Registering kernel %s: %p", this, kernel.name(), k);
    KernelMap::iterator i;
    i = kernels_.find(k);
    ASSERTION(i == kernels_.end());
    kernels_[k] = &kernel;
}

Mode &Mode::current()
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
    if(mode != NULL) mode->destroy();
    key.set(this);
    count_++;
}

void Mode::detach()
{
    Mode *mode = Mode::key.get();
    if(mode != NULL) mode->destroy();
    key.set(NULL);
}

void Mode::removeObject(memory::Object &obj)
{
    map_.remove(obj);
}


gmacError_t Mode::malloc(void **addr, size_t size, unsigned align)
{
    switchIn();
    error_ = acc_->malloc(addr, size, align);
    switchOut();
    return error_;
}

gmacError_t Mode::free(void *addr)
{
    switchIn();
    error_ = acc_->free(addr);
    switchOut();
    return error_;
}

gmacError_t Mode::copyToAccelerator(void *dev, const void *host, size_t size)
{
    TRACE(LOCAL,"Copy %p to device %p ("FMT_SIZE" bytes)", host, dev, size);
    switchIn();
    error_ = getContext().copyToAccelerator(dev, host, size);
    switchOut();
    return error_;
}

gmacError_t Mode::copyToHost(void *host, const void *dev, size_t size)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", dev , host, size);
    switchIn();
    error_ = getContext().copyToHost(host, dev, size);
    switchOut();
    return error_;
}

gmacError_t Mode::copyAccelerator(void *dst, const void *src, size_t size)
{
    switchIn();
    error_ = getContext().copyAccelerator(dst, src, size);
    switchOut();
    return error_;
}

gmacError_t Mode::memset(void *addr, int c, size_t size)
{
    switchIn();
    error_ = getContext().memset(addr, c, size);
    switchOut();
    return error_;
}

gmac::core::KernelLaunch &Mode::launch(const char *kernel)
{
    KernelMap::iterator i = kernels_.find(kernel);
    assert(i != kernels_.end());
    gmac::core::Kernel * k = i->second;
    switchIn();
    gmac::core::KernelLaunch &l = getContext().launch(*k);
    switchOut();

    return l;
}

gmacError_t Mode::sync()
{
    switchIn();
    error_ = contextMap_.sync();
    switchOut();
    return error_;
}

#ifndef USE_MMAP
bool Mode::requireUpdate(memory::Block &block)
{
	gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    return manager.requireUpdate(block);
}
#endif

// Nobody can enter GMAC until this has finished. No locks are needed
gmacError_t Mode::moveTo(Accelerator &acc)
{
    TRACE(LOCAL,"Moving mode from acc %d to %d", acc_->id(), acc.id());
    switchIn();
    gmacError_t ret = gmacSuccess;
    size_t free;
    size_t needed = map_.memorySize();
    acc_->memInfo(&free, NULL);

    if (needed > free) {
        return gmacErrorInsufficientAcceleratorMemory;
    }

    TRACE(LOCAL,"Releasing object memory in accelerator");
    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    map_.freeObjects(manager.protocol(), &gmac::memory::Protocol::toHost);

    TRACE(LOCAL,"Cleaning contexts");
    contextMap_.clean();

    TRACE(LOCAL,"Registering mode in new accelerator");
    acc_->unregisterMode(*this);
    acc_ = &acc;
    acc_->registerMode(*this);
    
    TRACE(LOCAL,"Reallocating objects");
    map_.reallocObjects(*this);

    TRACE(LOCAL,"Reloading mode");
    reload();

    //
    // \TODO What to do if there are many threads sharing the same mode!!
    //
    switchOut();

    return ret;
}

void Mode::memInfo(size_t *free, size_t *total)
{
    switchIn();
    acc_->memInfo(free, total);
    switchOut();
}

}}
