#include "Mode.h"
#include "IOBuffer.h"
#include "Accelerator.h"

#include <memory/Object.h>
#include <memory/Map.h>
#include <memory/Manager.h>
#include <gmac/init.h>

namespace gmac {

gmac::util::Private Mode::key;
unsigned Mode::next = 0;

Mode::Mode(Accelerator *acc) :
    _id(++next),
    _acc(acc),
#ifdef USE_VM
    _bitmap(new memory::vm::Bitmap()),
#endif
    _count(0)
{
    trace("Creating new memory map");
    _map = new memory::Map("ModeMemoryMap");
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
    delete _bitmap;
    delete _map;
    _acc->destroyMode(this); 
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

Mode *Mode::current()
{
    Mode *mode = static_cast<Mode *>(Mode::key.get());
    if(mode == NULL) mode = proc->create();
    gmac::util::Logger::ASSERTION(mode != NULL);
    return mode;
}

void Mode::attach()
{
    Mode *mode = static_cast<Mode *>(Mode::key.get());
    if(mode == this) return;
    if(mode != NULL) mode->destroy();
    key.set(this);
    _count++;
}

void Mode::detach()
{
    Mode *mode = static_cast<Mode *>(Mode::key.get());
    if(mode != NULL) mode->destroy();
    key.set(NULL);
}

gmacError_t Mode::malloc(void **addr, size_t size, unsigned align)
{
    switchIn();
    _error = _acc->malloc(addr, size, align);
    switchOut();
    return _error;
}

gmacError_t Mode::free(void *addr)
{
    switchIn();
    _error = _acc->free(addr);
    switchOut();
    return _error;
}

gmacError_t Mode::copyToDevice(void *dev, const void *host, size_t size)
{
    switchIn();
    _error = _context->copyToDevice(dev, host, size);
    switchOut();
    return _error;
}

gmacError_t Mode::copyToHost(void *host, const void *dev, size_t size)
{
    switchIn();
    _error = _context->copyToHost(host, dev, size);
    switchOut();
    return _error;
}

gmacError_t Mode::copyDevice(void *dst, const void *src, size_t size)
{
    switchIn();
    _error = _context->copyDevice(dst, src, size);
    switchOut();
    return _error;
}

gmacError_t Mode::memset(void *addr, int c, size_t size)
{
    switchIn();
    _error = _context->memset(addr, c, size);
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
    gmac::KernelLaunch *l  = _context->launch(k);
    switchOut();

    return l;
}

gmacError_t Mode::sync()
{
    switchIn();
    _error = _context->sync();
    switchOut();
    return _error;
}


#ifndef USE_MMAP
bool Mode::requireUpdate(memory::Block *block)
{
    return manager->requireUpdate(block);
}
#endif

}
