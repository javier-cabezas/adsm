#ifndef GMAC_MEMORY_BLOCKGROUP_INST_H_
#define GMAC_MEMORY_BLOCKGROUP_INST_H_

#include "core/address_space.h"

#include "block_state.h"

namespace __impl { namespace memory {

template<typename State>
inline void object_state<State>::modified_object()
{
    ASSERTION(bool(ownerShortcut_));

    ownerShortcut_->get_object_map().modified_objects();
#if 0
    AcceleratorMap::iterator i;
    for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        std::list<core::address_space *> aspaces = i->second;
        std::list<core::address_space *>::iterator j;
        for(j = i->second.begin(); j != i->second.end(); j++) {
            ObjectMap &map = (*j)->getAddressSpace();
            map.modified_objects();
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
object_state<State>::object_state(protocol &protocol,
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
    if (ownerShortcut_) ownerShortcut_->unmap(addr_, size_);
    if (shadow_ != NULL) memory_ops::unshadow(shadow_, size_);
    if (addr_ != NULL && hasUserMemory_ == false) memory_ops::unmap(addr_, size_);
    TRACE(LOCAL, "Destroying BlockGroup @ %p", addr_);
}

template<typename State>
inline accptr_t
object_state<State>::get_device_addr(const hostptr_t addr) const
{
    return deviceAddr_ + (addr - addr_);
}

template<typename State>
inline accptr_t
object_state<State>::get_device_addr() const
{
    return get_device_addr(addr_);
}

template<typename State>
inline core::address_space_ptr
object_state<State>::owner()
{
    ASSERTION(bool(ownerShortcut_));

    return ownerShortcut_;
}

template<typename State>
inline core::address_space_const_ptr
object_state<State>::owner() const
{
    ASSERTION(bool(ownerShortcut_));

    return ownerShortcut_;
}

template<typename State>
inline gmacError_t
object_state<State>::add_owner(core::address_space_ptr owner)
{
    ASSERTION(!ownerShortcut_);
    ownerShortcut_ = owner;

    gmacError_t ret = mallocAccelerator(*owner, addr_, size_, deviceAddr_);
    TRACE(LOCAL, "Add owner %p Object @ %p with device addr: %p", owner.get(), addr_, deviceAddr_.get_device_addr());

    return ret;
}

// TODO: move checks to DBC
template<typename State>
inline gmacError_t
object_state<State>::remove_owner(core::address_space_const_ptr owner)
{
    ASSERTION(ownerShortcut_ == owner);

    gmacError_t ret;
    hal::event_ptr evt = coherence_op(&protocol::remove_block, ret);
    ASSERTION(ret == gmacSuccess);
    evt = coherence_op(&protocol::unmap_from_device, ret);
    ASSERTION(ret == gmacSuccess);
    ownerShortcut_->unmap(addr_, size_);

    // Clean-up
    blocks_.clear();

    ownerShortcut_.reset();

    return gmacSuccess;
}

// TODO Receive a aspace
template<typename State>
inline gmacError_t
object_state<State>::map_to_device()
{
    ASSERTION(bool(ownerShortcut_));

    lock_write();

    // Allocate accelerator memory in the new aspace
    accptr_t newDeviceAddr(0);

    gmacError_t ret = mallocAccelerator(*ownerShortcut_, addr_, size_, newDeviceAddr);
    if (ret == gmacSuccess) {
        deviceAddr_ = newDeviceAddr;
    }

    unlock();

    return ret;
}

// TODO Receive a aspace
template <typename State>
inline gmacError_t
object_state<State>::unmap_from_device()
{
    gmacError_t ret;

    if (ownerShortcut_) {
        lock_write();
        // Remove blocks from the coherence domain
        hal::event_ptr evt = coherence_op(&protocol::unmap_from_device, ret);

        // Free accelerator memory
        if (ret == gmacSuccess) {
            ret = ownerShortcut_->unmap(addr_, size_);
            ASSERTION(ret == gmacSuccess, "Error unmapping object from accelerator");
        }
        unlock();
    }

    return ret;
}

}}

#endif
