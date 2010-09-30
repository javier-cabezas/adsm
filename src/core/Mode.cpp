#include "Accelerator.h"
#include "IOBuffer.h"
#include "Mode.h"
#include "Process.h"

#include "memory/Manager.h"
#include "memory/Object.h"
#include "memory/Protocol.h"

namespace gmac {

gmac::util::Private<Mode> Mode::key;
gmac::util::Private<Context> Mode::_context;

unsigned Mode::next = 0;

Mode::Mode(Process &proc, Accelerator &acc) :
    id_(++next),
    proc_(proc),
    acc_(&acc),
    _releasedObjects(true)
#ifdef USE_VM
    , _bitmap(new memory::vm::Bitmap())
#endif
    , _count(0)
{
    trace("Creating new memory map");
    _map = new memory::Map("ModeMemoryMap", *this);
}

Mode::~Mode()
{
    _count--;
    if(_count > 0)
        gmac::util::Logger::WARNING("Deleting in-use Execution Mode (%d)", _count);
    if(this == key.get()) key.set(NULL);
}

void
Mode::release()
{
#ifdef USE_VM
    delete _bitmap;
#endif
    delete _map;
    acc_->unregisterMode(*this); 
}
void Mode::kernel(gmacKernel_t k, Kernel * kernel)
{
    assertion(kernel != NULL);
    trace("CTX: %p Registering kernel %s: %p", this, kernel->name(), k);
    KernelMap::iterator i;
    i = kernels.find(k);
    assertion(i == kernels.end());
    kernels[k] = kernel;
}

Mode &Mode::current()
{
    Process &proc = Process::current();
    Mode *mode = Mode::key.get();
    if(mode == NULL) mode = proc.create();
    gmac::util::Logger::ASSERTION(mode != NULL);
    return *mode;
}

Context &Mode::currentContext()
{
    Context *context = Mode::_context.get();
    gmac::util::Logger::ASSERTION(context != NULL);
    return *context;
}

void Mode::attach()
{
    Mode *mode = Mode::key.get();
    if(mode == this) return;
    if(mode != NULL) mode->destroy();
    key.set(this);
    _count++;
}

void Mode::detach()
{
    Mode *mode = Mode::key.get();
    if(mode != NULL) mode->destroy();
    key.set(NULL);
}

gmacError_t Mode::malloc(void **addr, size_t size, unsigned align)
{
    switchIn();
    _error = acc_->malloc(addr, size, align);
    switchOut();
    return _error;
}

gmacError_t Mode::free(void *addr)
{
    switchIn();
    _error = acc_->free(addr);
    switchOut();
    return _error;
}

gmacError_t Mode::copyToDevice(void *dev, const void *host, size_t size)
{
    util::Logger::trace("Copy %p to device %p (%zd bytes)", host, dev, size);
    switchIn();
    _error = currentContext().copyToDevice(dev, host, size);
    switchOut();
    return _error;
}

gmacError_t Mode::copyToHost(void *host, const void *dev, size_t size)
{
    util::Logger::trace("Copy %p to host %p (%zd bytes)", dev , host, size);
    switchIn();
    _error = currentContext().copyToHost(host, dev, size);
    switchOut();
    return _error;
}

gmacError_t Mode::copyDevice(void *dst, const void *src, size_t size)
{
    switchIn();
    _error = currentContext().copyDevice(dst, src, size);
    switchOut();
    return _error;
}

gmacError_t Mode::memset(void *addr, int c, size_t size)
{
    switchIn();
    _error = currentContext().memset(addr, c, size);
    switchOut();
    return _error;
}

gmac::KernelLaunch *Mode::launch(const char *kernel)
{
    KernelMap::iterator i = kernels.find(kernel);
    assert(i != kernels.end());
    gmac::Kernel * k = i->second;
    assertion(k != NULL);
    switchIn();
    gmac::KernelLaunch *l = currentContext().launch(k);
    switchOut();

    return l;
}

gmacError_t Mode::sync()
{
    switchIn();
    _error = currentContext().sync();
    switchOut();
    return _error;
}

// TODO: remove this
extern gmac::memory::Manager *manager;

#ifndef USE_MMAP
bool Mode::requireUpdate(memory::Block &block)
{
    return manager->requireUpdate(block);
}
#endif

// Nobody can enter GMAC until this has finished. No locks are needed
gmacError_t Mode::moveTo(Accelerator &acc)
{
    trace("Moving to %d", acc_->id());
    switchIn();
    gmacError_t ret = gmacSuccess;
    size_t free;
    size_t needed = 0;
    acc_->memInfo(&free, NULL);
    gmac::memory::Map::const_iterator i;
    for(i = _map->begin(); i != _map->end(); i++) {
        gmac::memory::Object &object = *i->second;
        needed += object.size();
    }

    if (needed > free) {
        return gmacErrorInsufficientDeviceMemory;
    }

    for(i = _map->begin(); i != _map->end(); i++) {
        gmac::memory::Object &object = *i->second;
        manager->protocol().toHost(object);
        object.free();
    }

    acc_->unregisterMode(*this);
    delete Mode::_context.get();
    acc_ = &acc;
    acc_->registerMode(*this);
    newContext();

    for(i = _map->begin(); i != _map->end(); i++) {
        gmac::memory::Object &object = *i->second;
        object.realloc(*this);
    }

    CFatal(ret == gmacSuccess, "Error migrating context: not enough memory");

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

}
