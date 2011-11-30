#ifndef GMAC_MEMORY_BLOCKGROUP_INST_H_
#define GMAC_MEMORY_BLOCKGROUP_INST_H_

#include "core/address_space.h"

#include "block_state.h"

namespace __impl { namespace memory {

template<typename State>
inline void object_state<State>::modifiedObject()
{
    ASSERTION(ownerShortcut_ != NULL);

    ownerShortcut_->get_object_map().modifiedObjects();
#if 0
    AcceleratorMap::iterator i;
    for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        std::list<core::address_space *> aspaces = i->second;
        std::list<core::address_space *>::iterator j;
        for(j = i->second.begin(); j != i->second.end(); j++) {
            ObjectMap &map = (*j)->getAddressSpace();
            map.modifiedObjects();
        }
    }
#endif
}

static inline gmacError_t
mallocAccelerator(core::address_space &aspace, hostptr_t addr, size_t size, accptr_t &acceleratorAddr)
{
    acceleratorAddr = accptr_t();
    // Allocate accelerator memory
#ifdef USE_VM
    gmacError_t ret = aspace.map(, acceleratorAddr, addr, size, unsigned(SubBlockSize_));
    if(ret == gmacSuccess) {
        vm::Bitmap &bitmap = aspace.getBitmap();
        bitmap.registerRange(acceleratorAddr, size);
    }
#else
    gmacError_t ret = aspace.map(acceleratorAddr, addr, size);
#endif
    return ret;
}

template<typename State>
gmacError_t
object_state<State>::repopulateBlocks(core::address_space &aspace)
{
    abort();

    // Repopulate the block-set
    ptroff_t offset = 0;
    for (vector_block::iterator i = blocks_.begin(); i != blocks_.end(); i++) {
        block_state<State> &oldBlock = *dynamic_cast<block_state<State> *>(*i);
        block_state<State> *newBlock = new block_state<State>(oldBlock.getProtocol(),
                                                                addr_   + offset,
                                                                shadow_ + offset,
                                                                oldBlock.size(), oldBlock.getState());

        *i = newBlock;

        offset += ptroff_t(oldBlock.size());

        // Decrement reference count
        oldBlock.decRef();
    }

    return gmacSuccess;
}

template<typename State>
object_state<State>::object_state(protocol_interface &protocol,
                              hostptr_t hostAddr,
                              size_t size,
                              typename State::ProtocolState init,
                              gmacError_t &err) :
    object(protocol, hostAddr, size),
    shadow_(NULL),
    hasUserMemory_(hostAddr != NULL),
    deviceAddr_(0)
{
    err = gmacSuccess;

    // Allocate memory (if necessary)
    if(hostAddr == NULL) {
        addr_ = memory_ops::map(NULL, size, GMAC_PROT_READWRITE);
        if (addr_ == NULL) {
            err = gmacErrorMemoryAllocation;
            return;
        }
    }

    // Create a shadow mapping for the host memory
    shadow_ = memory_ops::shadow(addr_, size_);
    if (shadow_ == NULL) {
        err = gmacErrorMemoryAllocation;
        return;
    }

    hostptr_t mark = addr_;
    ptroff_t offset = 0;
    while(size > 0) {
        size_t blockSize = (size > BlockSize_) ? BlockSize_ : size;
        mark += blockSize;
        block_ptr block(new block_state<State>(*this, addr_ + offset,
                                                shadow_ + offset, blockSize, init));
        blocks_.push_back(block);
        size -= blockSize;
        offset += ptroff_t(blockSize);
		TRACE(LOCAL, "Creating BlockGroup @ %p : shadow @ %p ("FMT_SIZE" bytes) ", addr_, shadow_, blockSize);
    }
    
}


template<typename State>
object_state<State>::~object_state()
{
#if 0
    AcceleratorMap::iterator i;
    for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        std::list<core::address_space *> aspaces = i->second;
        if (aspaces.size() > 0) {
            aspaces.front()->unmap(addr_, size_);
        }
#ifdef USE_VM
        std::list<core::address_space *>::iterator j;
        for (j = i->second->begin(); j != i->second->end; j++) {
            vm::Bitmap &bitmap = (*j)->getBitmap();
            bitmap.unregisterRange(acceleratorAddr_, size_);
        }
#endif
    }
#endif
    if (ownerShortcut_ != NULL) ownerShortcut_->unmap(addr_, size_);
    if (shadow_ != NULL) memory_ops::unshadow(shadow_, size_);
    if (addr_ != NULL && hasUserMemory_ == false) memory_ops::unmap(addr_, size_);
    TRACE(LOCAL, "Destroying BlockGroup @ %p", addr_);
}

template<typename State>
inline accptr_t
object_state<State>::get_device_addr(const hostptr_t addr) const
{
    return deviceAddr_ + (addr - addr_);
#if 0
    map_aspace::const_iterator m;
    if (owners_.size() == 1) {
        m = owners_.begin();
        ret = m->second + (addr - this->addr_);
    } else {
        m = owners_.find(&current);
        if (m != owners_.end()) {
            ret = m->second + (addr - this->addr_);
        }
    }

    //StateBlock<State>::unlock();
    return ret;
#endif
}

template<typename State>
inline accptr_t
object_state<State>::get_device_addr() const
{
    return get_device_addr(addr_);
#if 0
    map_aspace::const_iterator m;
    if (owners_.size() == 1) {
        m = owners_.begin();
        ret = m->second + (addr - this->addr_);
    } else {
        m = owners_.find(&current);
        if (m != owners_.end()) {
            ret = m->second + (addr - this->addr_);
        }
    }

    //StateBlock<State>::unlock();
    return ret;
#endif
}

template<typename State>
inline core::address_space_ptr
object_state<State>::owner()
{
    ASSERTION(ownerShortcut_);

    return ownerShortcut_;
#if 0
    core::address_space *ret;
    lock_read();
    if (owners_.size() == 1) {
        ret = ownerShortcut_;
    } else {
        map_aspace::const_iterator m;
        m = owners_.find(&current);
        if (m == owners_.end()) {
            ret = owners_.begin()->first;
        } else {
            ret = m->first;
        }
    }
    unlock();
    return *ret;
#endif
}

template<typename State>
inline core::address_space_const_ptr
object_state<State>::owner() const
{
    ASSERTION(ownerShortcut_);

    core::address_space_const_ptr ret(ownerShortcut_);

    return ret;
}

template<typename State>
inline gmacError_t
object_state<State>::addOwner(core::address_space_ptr owner)
{
    ASSERTION(!ownerShortcut_);
    ownerShortcut_ = owner;

    gmacError_t ret = mallocAccelerator(*owner, addr_, size_, deviceAddr_);
    TRACE(LOCAL, "Add owner %p Object @ %p with device addr: %p", owner.get(), addr_, deviceAddr_.get_device_addr());

    return ret;
#if 0
    accptr_t acceleratorAddr = accptr_t(0);

    gmacError_t ret = mallocAccelerator(aspace, addr_, size_, acceleratorAddr);
    if (ret != gmacSuccess) return ret;

    lock_write();

    AcceleratorMap::iterator it = acceleratorAddr_.find(acceleratorAddr);
    if (it == acceleratorAddr_.end()) {
        acceleratorAddr_.insert(AcceleratorMap::value_type(acceleratorAddr, std::list<core::address_space *>()));
        AcceleratorMap::iterator it = acceleratorAddr_.find(acceleratorAddr);
        it->second.push_back(&aspace);
    } else {
        it->second.push_back(&aspace);
    }

    ASSERTION(owners_.find(&aspace) == owners_.end());
    owners_.insert(map_aspace::value_type(&aspace, addr));

    if (owners_.size() == 1) {
        ownerShortcut_ = &aspace;
    } else {
        ownerShortcut_ = NULL;
    }
    unlock();
    return gmacSuccess;
#endif
}

// TODO: move checks to DBC
template<typename State>
inline gmacError_t
object_state<State>::removeOwner(core::address_space_const_ptr owner)
{
    ASSERTION(ownerShortcut_ == owner);

    gmacError_t ret = coherenceOp(&protocol_interface::deleteBlock);
    ASSERTION(ret == gmacSuccess);
    ret = coherenceOp(&protocol_interface::unmapFromAccelerator);
    ASSERTION(ret == gmacSuccess);
    ownerShortcut_->unmap(addr_, size_);

#if 0
    acceleratorAddr_.clear();
#endif

    // Clean-up
    blocks_.clear();

    ownerShortcut_.reset();

    return gmacSuccess;
#if 0
    lock_write();

    TRACE(LOCAL, "Remove owner %p Object @ %p: %u -> %u", &aspace, addr_, owners_.size(), owners_.size() - 1);

    ASSERTION(owners_.size() > 0);

    map_aspace::iterator m;
    m = owners_.find(&aspace);
    ASSERTION(m != owners_.end());
    owners_.erase(m);

    if (owners_.size() == 0) {
        ASSERTION(acceleratorAddr_.size() == 1);

        gmacError_t ret = coherenceOp(&protocol_interface::deleteBlock);
        ASSERTION(ret == gmacSuccess);
        ret = coherenceOp(&protocol_interface::unmapFromAccelerator);
        ASSERTION(ret == gmacSuccess);
        ownerShortcut_->unmap(addr_, size_);

        acceleratorAddr_.clear();

        // Clean-up
        vector_block::iterator i;
        for(i = blocks_.begin(); i != blocks_.end(); i++) {
            i->second->decRef();
        }
        blocks_.clear();

        ownerShortcut_ = NULL;
    } else {
        AcceleratorMap::iterator i;
        bool ownerFound = false;
        for (i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
            std::list<core::address_space *> &list = i->second;
            std::list<core::address_space *>::iterator j = std::find(list.begin(), list.end(), &aspace);
            if (j != list.end()) {
                list.erase(j);
                if (list.size() == 0) {
                    acceleratorAddr_.erase(i);
                    aspace.unmap(addr_, size_);
                }
                ownerFound = true;
                break;
            }
        }
        ASSERTION(ownerFound == true);

#if 0
        for (vector_block::iterator j = blocks_.begin(); j != blocks_.end(); j++) {
            block_state<State> &block = dynamic_cast<block_state<State> &>(*j->second);
            block.removeOwner(aspace);
        }
#endif

        if (owners_.size() == 1) {
            ASSERTION(acceleratorAddr_.size() == 1);
            i = acceleratorAddr_.begin();
            std::list<core::address_space *> &list = i->second;
            ASSERTION(list.size() == 1);
            ownerShortcut_ = list.front();
        }
    }

    unlock();
    return gmacSuccess;
#endif
}

// TODO Receive a aspace
template<typename State>
inline gmacError_t
object_state<State>::mapToAccelerator()
{
    ASSERTION(ownerShortcut_);

    lock_write();

    // Allocate accelerator memory in the new aspace
    accptr_t newDeviceAddr(0);

    gmacError_t ret = mallocAccelerator(*ownerShortcut_, addr_, size_, newDeviceAddr);
    if (ret == gmacSuccess) {
        deviceAddr_ = newDeviceAddr;
    }

    unlock();

    return ret;
#if 0
    if (ret == gmacSuccess) {
        ASSERTION(acceleratorAddr_.size() == 1);
        acceleratorAddr_.clear();
        acceleratorAddr_.insert(AcceleratorMap::value_type(newAcceleratorAddr, std::list<core::address_space *>()));
        AcceleratorMap::iterator it = acceleratorAddr_.find(newAcceleratorAddr);
        it->second.push_back(ownerShortcut_);

        // Recreate accelerator blocks
        repopulateBlocks(*ownerShortcut_);
        // Add blocks to the coherence domain
        ret = coherenceOp(&protocol_interface::mapToAccelerator);

        deviceAddr_ = newDeviceAddr_;
    }

    unlock();

    gmacError_t ret;

    lock_write();

    if (owners_ == 1) {
        // Allocate accelerator memory in the new aspace
        accptr_t newAcceleratorAddr(0);

        ret = mallocAccelerator(*ownerShortcut_, addr_, size_, newAcceleratorAddr);

        if (ret == gmacSuccess) {
            ASSERTION(acceleratorAddr_.size() == 1);
            acceleratorAddr_.clear();
            acceleratorAddr_.insert(AcceleratorMap::value_type(newAcceleratorAddr, std::list<core::address_space *>()));
            AcceleratorMap::iterator it = acceleratorAddr_.find(newAcceleratorAddr);
            it->second.push_back(ownerShortcut_);
            
            // Recreate accelerator blocks
            repopulateBlocks(newAcceleratorAddr, *ownerShortcut_);
            // Add blocks to the coherence domain
            ret = coherenceOp(&protocol_interface::mapToAccelerator);
        }
    } else {
        // Not supported for now
        return gmacErrorFeatureNotSupported;
    }

    unlock();
    return ret;
#endif
}

// TODO Receive a aspace
template <typename State>
inline gmacError_t
object_state<State>::unmapFromAccelerator()
{
    gmacError_t ret = gmacSuccess;

    if (ownerShortcut_) {
        lock_write();
        // Remove blocks from the coherence domain
        ret = coherenceOp(&protocol_interface::unmapFromAccelerator);

        // Free accelerator memory
        if (ret == gmacSuccess) {
            ret = ownerShortcut_->unmap(addr_, size_);
            ASSERTION(ret == gmacSuccess, "Error unmapping object from accelerator");
        }
        unlock();
    }

    return ret;
#if 0
    // Not supported for now
    if (owners_.size() == 1) {
        // Remove blocks from the coherence domain
        ret = coherenceOp(&protocol_interface::unmapFromAccelerator);

        // Free accelerator memory
        if (ret == gmacSuccess) {
            ret = ownerShortcut_->unmap(addr_, size_);
            ASSERTION(ret == gmacSuccess, "Error unmapping object from accelerator");
        }
    } else {
        ret = gmacErrorFeatureNotSupported;
    }
    unlock();
    return ret;
#endif
}

}}

#endif
