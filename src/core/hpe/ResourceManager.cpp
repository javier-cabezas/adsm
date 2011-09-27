#include "core/hpe/ResourceManager.h"

#include "core/hpe/Mode.h"

namespace __impl { namespace core { namespace hpe {

ResourceManager::ResourceManager() :
    core::ResourceManager()
{
}

ResourceManager::~ResourceManager()
{
}

gmacError_t
ResourceManager::hostAlloc(core::Mode &mode, hostptr_t &addr, size_t size)
{
    return mode.hostAlloc(addr, size);
}

gmacError_t
ResourceManager::hostFree(core::Mode &mode, hostptr_t addr)
{
    return mode.hostFree(addr);
}

accptr_t
ResourceManager::hostMapAddr(core::Mode &mode, const hostptr_t addr)
{
    return mode.hostMapAddr(addr);
}

gmacError_t
ResourceManager::map(core::Mode &mode, accptr_t &dst, hostptr_t src, size_t count, unsigned align)
{
    return mode.map(dst, src, count, align);
}

gmacError_t
ResourceManager::unmap(core::Mode &mode, hostptr_t addr, size_t count)
{
    return mode.unmap(addr, count);
}

gmacError_t
ResourceManager::copyToAccelerator(core::Mode &mode, accptr_t acc, const hostptr_t host, size_t count)
{
    return mode.copyToAccelerator(acc, host, count);
}

gmacError_t
ResourceManager::copyToHost(core::Mode &mode, hostptr_t host, const accptr_t acc, size_t count)
{
    return mode.copyToHost(host, acc, count);
}

gmacError_t
ResourceManager::copyAccelerator(core::Mode &mode, accptr_t dst, const accptr_t src, size_t count)
{
    return mode.copyAccelerator(dst, src, count);
}

IOBuffer &
ResourceManager::createIOBuffer(core::Mode &mode, size_t count, GmacProtection prot)
{
    return mode.createIOBuffer(count, prot);
}

void
ResourceManager::destroyIOBuffer(core::Mode &mode, core::IOBuffer &buffer)
{
    return mode.destroyIOBuffer(buffer);
}

gmacError_t
ResourceManager::bufferToAccelerator(core::Mode &mode, accptr_t dst, core::IOBuffer &buffer, size_t count, size_t off)
{
    return mode.bufferToAccelerator(dst, buffer, count, off);
}

gmacError_t
ResourceManager::acceleratorToBuffer(core::Mode &mode, core::IOBuffer &buffer, const accptr_t dst, size_t count, size_t off)
{
    return mode.acceleratorToBuffer(buffer, dst, count, off);
}

gmacError_t
ResourceManager::memset(core::Mode &mode, accptr_t addr, int c, size_t size)
{
    return mode.memset(addr, c, size);
}

}}}
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
