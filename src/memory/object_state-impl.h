#ifndef GMAC_MEMORY_BLOCKGROUP_INST_H_
#define GMAC_MEMORY_BLOCKGROUP_INST_H_

namespace __impl { namespace memory {

template<typename ProtocolTraits>
inline void object_state<ProtocolTraits>::modified_object()
{
    ASSERTION(bool(ownerShortcut_));

    ownerShortcut_->get_object_map().modified_objects();
}

static inline gmacError_t
malloc_accelerator(address_space &aspace, host_ptr addr, size_t size, hal::ptr &acceleratorAddr)
{
    acceleratorAddr = hal::ptr();
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

template<typename ProtocolTraits>
gmacError_t
object_state<ProtocolTraits>::repopulate_blocks(address_space &aspace)
{
    FATAL("Not implemented");

    // Repopulate the block-set
    ptroff_t offset = 0;
    for (const_locking_iterator i = begin(); i != end(); ++i) {
        typename ProtocolTraits::block &oldBlock = *dynamic_cast<typename ProtocolTraits::block *>(*i);
        typename ProtocolTraits::block *newBlock = new typename ProtocolTraits::block(oldBlock.getProtocol(),
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

template<typename ProtocolTraits>
object_state<ProtocolTraits>::object_state(protocol &protocol,
                                           host_ptr hostAddr, size_t size,
                                           int flagsHost, int flagsDevice,
                                           typename ProtocolTraits::State init,
                                           gmacError_t &err) :
    gmac::memory::object(protocol, hostAddr, size, flagsHost, flagsDevice),
    shadow_(NULL),
    hasUserMemory_(hostAddr != NULL),
    deviceAddr_()
{
    err = gmacSuccess;

    // Generate host address
    if (hostAddr == NULL) {
        // Create new host address
        if (flagsHost == GMAC_MAP_NONE) {
            // Create balloon to identify the object
            addr_ = memory_ops::map(NULL, size, GMAC_PROT_NONE);
        } else {
            // Allocate new (accessible) memory
            if (flagsHost & GMAC_MAP_READWRITE) {
                addr_ = memory_ops::map(NULL, size, GMAC_PROT_READWRITE);
            } else if (flagsHost & GMAC_MAP_READ) {
                addr_ = memory_ops::map(NULL, size, GMAC_PROT_READ);
            } else /* (flagsHost & GMAC_MAP_WRITE) */ {
                addr_ = memory_ops::map(NULL, size, GMAC_PROT_WRITE);
            }
        }

        // Check if host address has been correctly generated
        if (addr_ == NULL) {
            err = gmacErrorMemoryAllocation;
            return;
        }
    } else {
        // Map host address
        if (flagsHost & GMAC_MAP_READWRITE) {
            addr_ = memory_ops::map(addr_, size, GMAC_PROT_READWRITE);
        } else if (flagsHost & GMAC_MAP_READ) {
            addr_ = memory_ops::map(addr_, size, GMAC_PROT_READ);
        } else /* (flagsHost & GMAC_MAP_WRITE) */ {
            addr_ = memory_ops::map(addr_, size, GMAC_PROT_WRITE);
        }
    }

    // Create shadow copy of the object if necessary
    if (flagsHost   != GMAC_MAP_NONE &&
        flagsDevice != GMAC_MAP_NONE) {
        // Create a shadow mapping for the host memory
        shadow_ = memory_ops::shadow(addr_, size_);
        if (shadow_ == NULL) {
            err = gmacErrorMemoryAllocation;
            return;
        }
    }

    ptroff_t offset = 0;
    while (size > 0) {
        size_t blockSize = (size > BlockSize_) ? BlockSize_ : size;
        typename ProtocolTraits::block_ptr block(new typename ProtocolTraits::block(*this, offset,
                                                                                    blockSize, init));
        blocks_.push_back(block);

		TRACE(LOCAL, "Creating BlockGroup @ %p : shadow @ %p ("FMT_SIZE" bytes) ",
                     addr_ + offset,
                     shadow_ + offset, blockSize);
        size -= blockSize;
        offset += ptroff_t(blockSize);
    }
}

template<typename ProtocolTraits>
object_state<ProtocolTraits>::~object_state()
{
    if (ownerShortcut_) ownerShortcut_->unmap(addr_, size_);
    if (shadow_ != NULL) memory_ops::unshadow(shadow_, size_);
    if (addr_ != NULL && hasUserMemory_ == false) memory_ops::unmap(addr_, size_);
    TRACE(LOCAL, "Destroying BlockGroup @ %p", addr_);
}

template<typename ProtocolTraits>
inline hal::ptr
object_state<ProtocolTraits>::get_device_addr(host_ptr addr)
{
    return deviceAddr_ + (addr - addr_);
}

template<typename ProtocolTraits>
inline hal::const_ptr
object_state<ProtocolTraits>::get_device_const_addr(host_const_ptr addr) const
{
    hal::ptr ret = deviceAddr_ + (addr - addr_);
    return hal::const_ptr(ret);
}

template<typename ProtocolTraits>
inline hal::ptr
object_state<ProtocolTraits>::get_device_addr()
{
    return get_device_addr(addr_);
}

template<typename ProtocolTraits>
inline hal::const_ptr
object_state<ProtocolTraits>::get_device_const_addr() const
{
    return get_device_const_addr(addr_);
}

template<typename ProtocolTraits>
inline address_space_ptr
object_state<ProtocolTraits>::get_owner()
{
    ASSERTION(bool(ownerShortcut_));

    return ownerShortcut_;
}

template<typename ProtocolTraits>
inline address_space_const_ptr
object_state<ProtocolTraits>::get_owner() const
{
    ASSERTION(bool(ownerShortcut_));

    return ownerShortcut_;
}

template<typename ProtocolTraits>
inline gmacError_t
object_state<ProtocolTraits>::add_owner(address_space_ptr owner)
{
    ASSERTION(!ownerShortcut_);
    ownerShortcut_ = owner;

    gmacError_t ret = malloc_accelerator(*owner, addr_, size_, deviceAddr_);
    TRACE(LOCAL, "Add owner %p Object @ %p with device addr: %p", owner.get(), addr_, deviceAddr_.get_device_addr());

    return ret;
}

// TODO: move checks to DBC
template<typename ProtocolTraits>
inline gmacError_t
object_state<ProtocolTraits>::remove_owner(address_space_const_ptr owner)
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
template<typename ProtocolTraits>
inline gmacError_t
object_state<ProtocolTraits>::map_to_device()
{
    ASSERTION(bool(ownerShortcut_));

    // Allocate accelerator memory in the new aspace
    hal::ptr newDeviceAddr;

    gmacError_t ret = malloc_accelerator(*ownerShortcut_, addr_, size_, newDeviceAddr);
    if (ret == gmacSuccess) {
        deviceAddr_ = newDeviceAddr;
    }

    return ret;
}

// TODO Receive a aspace
template <typename ProtocolTraits>
inline gmacError_t
object_state<ProtocolTraits>::unmap_from_device()
{
    gmacError_t ret;

    if (ownerShortcut_) {
        // Remove blocks from the coherence domain
        hal::event_ptr evt = coherence_op(&protocol::unmap_from_device, ret);

        // Free accelerator memory
        if (ret == gmacSuccess) {
            ret = ownerShortcut_->unmap(addr_, size_);
            ASSERTION(ret == gmacSuccess, "Error unmapping object from accelerator");
        }
    }

    return ret;
}

}}

#endif
