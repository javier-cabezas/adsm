#include "core/hpe/ResourceManager.h"

#include "core/hpe/Mode.h"
#include "core/hpe/VirtualDeviceTable.h"
#include "core/hpe/Thread.h"

namespace __impl { namespace core { namespace hpe {

ResourceManager::ResourceManager(Process &proc) :
    core::ResourceManager(),
    proc_(proc)
{
}

ResourceManager::~ResourceManager()
{
}

gmacError_t
ResourceManager::newAddressSpace(GmacAddressSpaceId &aSpaceId, unsigned accId)
{
    AddressSpace *aSpace = new AddressSpace("", proc_, proc_.getAccelerator(accId));

    aSpaceId = aSpace->getId();

    return gmacSuccess;
}

gmacError_t
ResourceManager::deleteAddressSpace(GmacAddressSpaceId aSpaceId)
{
    gmacError_t ret = gmacSuccess;

    AddressSpaceMap::iterator it;
    it = aSpaceMap_.find(aSpaceId);

    if (it != aSpaceMap_.end()) {
        aSpaceMap_.erase(it);
    } else {
        ret = gmacErrorInvalidValue;
    }

    return ret;
}

AddressSpace *
ResourceManager::getAddressSpace(GmacAddressSpaceId aSpaceId)
{
    AddressSpace *ret = NULL;

    AddressSpaceMap::iterator it;
    it = aSpaceMap_.find(aSpaceId);

    if (it != aSpaceMap_.end()) {
        ret = it->second;
    }

    return ret;
}

gmacError_t
ResourceManager::newVirtualDevice(GmacVirtualDeviceId &vDeviceId, GmacAddressSpaceId aSpaceId)
{
    gmacError_t ret = gmacSuccess;

    AddressSpaceMap::iterator it;
    it = aSpaceMap_.find(aSpaceId);

    if (it != aSpaceMap_.end()) {
        AddressSpace &aSpace = *it->second;
        Mode *mode = proc_.createMode(aSpace.getAccelerator().id());
        VirtualDeviceTable &vDeviceTable = Thread::getCurrentVirtualDeviceTable();
        ret = vDeviceTable.addVirtualDevice(mode->getId(), *mode);
    } else {
        ret = gmacErrorInvalidValue;
    }

    return ret;
}

gmacError_t
ResourceManager::deleteVirtualDevice(GmacVirtualDeviceId vDeviceId)
{
    gmacError_t ret = gmacSuccess;

    VirtualDeviceTable &vDeviceTable = Thread::getCurrentVirtualDeviceTable();
    ret = vDeviceTable.removeVirtualDevice(vDeviceId);

    return ret;
}

Mode *
ResourceManager::getVirtualDevice(GmacVirtualDeviceId vDeviceId)
{
    Mode *ret = NULL;

    VirtualDeviceTable &vDeviceTable = Thread::getCurrentVirtualDeviceTable();
    ret = vDeviceTable.getVirtualDevice(vDeviceId);

    return ret;
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
