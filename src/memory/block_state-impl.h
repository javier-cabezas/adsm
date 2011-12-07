#ifndef GMAC_MEMORY_GENERICBLOCK_INST_H_
#define GMAC_MEMORY_GENERICBLOCK_INST_H_

#include <algorithm>

#include "core/address_space.h"

namespace __impl { namespace memory {

template<typename State>
inline
block_state<State>::block_state(object_state<State> &parent,
                                  hostptr_t hostAddr,
                                  hostptr_t shadowAddr,
                                  size_t size,
                                  typename State::ProtocolState init) :
    gmac::memory::block(hostAddr, shadowAddr, size),
    State(init),
    parent_(parent)
{
}

template<typename State>
inline
block_state<State>::~block_state()
{
}

#if 0
template<typename State>
void
block_state<State>::addOwner(core::address_space &aspace, accptr_t addr)
{
    lock();

    AcceleratorAddrMap::iterator it = acceleratorAddr_.find(addr);

    TRACE(LOCAL, "Adding owner for address for %u:%p @ Context %p", addr.getPAddressSpace(), addr.get(), &aspace);
    if (it == acceleratorAddr_.end()) {
        TRACE(LOCAL, "Adding new address for %u:%p @ Context %p", addr.getPAddressSpace(), addr.get(), &aspace);
        acceleratorAddr_.insert(AcceleratorAddrMap::value_type(addr, std::list<core::address_space *>()));
        AcceleratorAddrMap::iterator it = acceleratorAddr_.find(addr);
        it->second.push_back(&aspace);

        if(protocol_.needUpdate(*this) == true &&
           acceleratorAddr_.size() > 1) {
            gmacError_t ret = aspace.copyToAccelerator(addr, shadow_, size_);
            ASSERTION(ret == gmacSuccess);
        }
    } else {
        it->second.push_back(&aspace);
    }

    ASSERTION(owners_.find(&aspace) == owners_.end());
    owners_.insert(AddressSpaceMap::value_type(&aspace, addr));

    if (owners_.size() == 1) {
        ownerShortcut_ = &aspace;
    }

    unlock();
}

// TODO: return error!
template<typename State>
void
block_state<State>::removeOwner(core::address_space &aspace)
{
    lock();

    AcceleratorAddrMap::iterator a;
    for (a = acceleratorAddr_.begin(); a != acceleratorAddr_.end(); a++) {
        std::list<core::address_space *> &list = a->second;
        std::list<core::address_space *>::iterator j = std::find(list.begin(), list.end(), &aspace);
        if (j != list.end()) {
            list.erase(j);
            if (list.size() == 0) acceleratorAddr_.erase(a);
            goto done_addr;
        }
    }
    FATAL("Mode NOT found!");
done_addr:

    AddressSpaceMap::iterator m;
    m = owners_.find(&aspace);
    ASSERTION(m != owners_.end());
    owners_.erase(m);

    unlock();
}
#endif

template<typename State>
inline core::address_space_ptr
block_state<State>::owner() const
{
    return parent_.owner();
#if 0
    core::address_space *ret;
    ASSERTION(owners_.size() > 0);

    if (owners_.size() == 1) {
        ret = ownerShortcut_;
    } else {
        AddressSpaceMap::const_iterator m;
        m = owners_.find(&current);
        if (m == owners_.end()) {
            ret = owners_.begin()->first;
        } else {
            ret = m->first;
        }
    }
    return *ret;
#endif
}

template<typename State>
inline accptr_t
block_state<State>::get_device_addr(const hostptr_t addr) const
{
    return parent_.get_device_addr(addr);
#if 0
    accptr_t ret = accptr_t(0);

    //lock();

    AddressSpaceMap::const_iterator m;
    if (owners_.size() == 1) {
        m = owners_.begin();
        ret = m->second + (addr - this->addr_);
    } else {
        m = owners_.find(&current);
        if (m != owners_.end()) {
            ret = m->second + (addr - this->addr_);
        }
    }

    //unlock();
    return ret;
#endif
}

template<typename State>
inline accptr_t
block_state<State>::get_device_addr() const
{
    return get_device_addr(this->addr_);
}

template<typename State>
inline hal::event_t
block_state<State>::to_host(unsigned blockOff, size_t count, gmacError_t &err)
{
    hal::event_t ret;

    err = parent_.owner()->copy(hal::ptr_t(this->shadow_ + blockOff),
                                get_device_addr() + blockOff,
                                count);
#if 0
    // Fast path
    if (owners_.size() == 1) {
        AddressSpaceMap::const_iterator m;
        ret = parent_.owner().copyToHost(this->shadow_ + blockOff,
                                         get_device_addr() + blockOff,
                                         count);
    } else { // TODO Implement this path
        ret = gmacSuccess;
    }
#endif

    return ret;
}

template<typename State>
inline hal::event_t
block_state<State>::to_accelerator(unsigned blockOff, size_t count, gmacError_t &err)
{
    hal::event_t ret;
    ret = parent_.owner()->copy_async(get_device_addr() + blockOff,
                                      hal::ptr_t(shadow_ + blockOff),
                                      count, err);
#if 0
    // Fast path
    if (owners_.size() == 1) {
        AddressSpaceMap::const_iterator m;
        ret = ownerShortcut_->copyToAccelerator(m->second + blockOff, shadow_ + blockOff, count);
    } else {
        AcceleratorAddrMap::const_iterator a;
        for(a = acceleratorAddr_.begin(); a != acceleratorAddr_.end(); a++) {
            const std::list<core::address_space *> &list = a->second;
            ASSERTION(list.size() > 0);
            core::address_space *aspace = list.front();
            ret = this->resourceManager_.copyToAccelerator(*aspace, a->first + blockOff, shadow_ + blockOff, count);
            if(ret != gmacSuccess) break;
        }
    }
#endif

    return ret;
}

#if 0
template<typename State>
inline gmacError_t
block_state<State>::copyFromBuffer(size_t blockOff, core::io_buffer &buffer,
                                    size_t bufferOff, size_t size, Destination dst) const
{
    gmacError_t ret = gmacSuccess;

    switch (dst) {
    case HOST:
        ::memcpy(shadow_ + blockOff, buffer.addr() + bufferOff, size);
        break;

    case ACCELERATOR:
        ret = parent_.owner().copy(get_device_addr() + ptroff_t(blockOff),
                                   buffer, bufferOff, size);
        break;
#if 0
        if (owners_.size() == 1) { // Fast path
            AddressSpaceMap::const_iterator m;
            m = owners_.begin();
            ret = ownerShortcut_->bufferToAccelerator(m->second + ptroff_t(blockOff), buffer, size, bufferOff);
        } else {
            AcceleratorAddrMap::const_iterator i;
            for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
                const std::list<core::address_space *> &list = i->second;
                ASSERTION(list.size() > 0);
                core::address_space *aspace = list.front();
                ret = aspace->bufferToAccelerator(i->first + ptroff_t(blockOff), buffer, size, bufferOff);
                if (ret != gmacSuccess) break;
            }
        }
        break;
#endif
    }

    return ret;
}

template<typename State>
inline gmacError_t
block_state<State>::copyToBuffer(core::io_buffer &buffer, size_t bufferOff,
                                  size_t blockOff, size_t size, Source src) const
{
    gmacError_t ret = gmacSuccess;
    switch (src) {
    case HOST:
        ::memcpy(buffer.addr() + bufferOff, shadow_ + blockOff, size);
        break;
    case ACCELERATOR:
        ret = parent_.owner().copy(buffer,
                                   bufferOff,
                                   get_device_addr() + ptroff_t(blockOff),
                                   size);
        break;
#if 0
        if (owners_.size() == 1) { // Fast path
            AddressSpaceMap::const_iterator m;
            m = owners_.begin();
            ret = ownerShortcut_->acceleratorToBuffer(buffer, m->second + ptroff_t(blockOff), size, bufferOff);
        } else {
            ret = gmacErrorFeatureNotSupported;
        }
        break;
#endif
    }

    return ret;
}

template <typename State>
gmacError_t
block_state<State>::memset(int v, size_t size, size_t blockOffset, Destination dst) const
{
    gmacError_t ret = gmacSuccess;
    if (dst == HOST) {
    } else  {
        ret = parent_.owner().memset(get_device_addr() + ptroff_t(blockOffset),
                                     v, size);
#if 0
        if (owners_.size() == 1) { // Fast path
            AddressSpaceMap::const_iterator m;
            m = owners_.begin();
            ret = ownerShortcut_->memset(m->second + ptroff_t(blockOffset), v, size);
        } else {
            AcceleratorAddrMap::const_iterator i;
            for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
                const std::list<core::address_space *> &list = i->second;
                ASSERTION(list.size() > 0);
                core::address_space *aspace = list.front();
                ret = this->resourceManager_.memset(*aspace, i->first + ptroff_t(blockOffset), v, size);
                if(ret != gmacSuccess) break;
            }
        }
#endif
    }
    return ret;
}
#endif

}}

#endif
