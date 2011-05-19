#ifndef GMAC_MEMORY_SHAREDOBJECT_IMPL_H_
#define GMAC_MEMORY_SHAREDOBJECT_IMPL_H_

#include "memory/SharedBlock.h"

#include "core/Process.h"

namespace __impl { namespace memory {

template<typename State>
void SharedObject<State>::validate()
{
    if(owner_ != NULL) owner_->validateObjects();
}

template<typename State>
accptr_t SharedObject<State>::allocAcceleratorMemory(core::Mode &mode, hostptr_t addr, size_t size)
{
    accptr_t acceleratorAddr(0);
    // Allocate accelerator memory
#ifdef USE_VM
    gmacError_t ret = 
        mode.map(acceleratorAddr, addr, size, unsigned(SubBlockSize_));
#else
    gmacError_t ret = 
        mode.map(acceleratorAddr, addr, size);
#endif
    if(ret == gmacSuccess) {
#ifdef USE_VM
        vm::Bitmap &bitmap = mode.getBitmap();
        bitmap.registerRange(acceleratorAddr, size);
#endif
        return acceleratorAddr;
    } else {
        return accptr_t(0);
    }
}

template<typename State>
gmacError_t SharedObject<State>::repopulateBlocks(accptr_t accPtr, core::Mode &mode)
{
    // Repopulate the block-set
    ptroff_t offset = 0;
    for(BlockMap::iterator i = blocks_.begin(); i != blocks_.end(); i++) {
        SharedBlock<State> &oldBlock = *dynamic_cast<SharedBlock<State> *>(i->second);
        SharedBlock<State> *newBlock = new SharedBlock<State>(oldBlock.getProtocol(), mode,
                                                      addr_   + offset,
                                                      shadow_ + offset,
                                                      accPtr  + offset,
                                                      oldBlock.size(), oldBlock.getState());

        i->second = newBlock;

        offset += ptroff_t(oldBlock.size());

        // Decrement reference count
        oldBlock.release();
    }

    return gmacSuccess;
}

template<typename State>
SharedObject<State>::SharedObject(Protocol &protocol, core::Mode &owner, hostptr_t hostAddr, size_t size, typename State::ProtocolState init) :
    Object(hostAddr, size),
    acceleratorAddr_(0),
	owner_(&owner)
{
    addr_ = NULL;

    // Allocate memory (if necessary)
    if(hostAddr == NULL) {
        addr_ = Memory::map(NULL, size, GMAC_PROT_READWRITE);
        valid_ = (addr_ != NULL);
        if(valid_ == false) return;
    }
    else {
        addr_ = hostAddr;
    }

    if (valid_ == true) {
        // Allocate accelerator memory
        acceleratorAddr_ = allocAcceleratorMemory(owner, addr_, size);
        valid_ = (acceleratorAddr_ != 0);
    }

    // Free allocated memory if there has been an error
    if (valid_ == false && hostAddr == NULL) {
        Memory::unmap(addr_, size);
    }

    if (valid_ == true) {
        // Create a shadow mapping for the host memory
        // TODO: check address
        shadow_ = hostptr_t(Memory::shadow(addr_, size_));
        // Populate the block-set
        hostptr_t mark = addr_;
        ptroff_t offset = 0;
        while(size > 0) {
            size_t blockSize = (size > BlockSize_) ? BlockSize_ : size;
            mark += blockSize;
            blocks_.insert(BlockMap::value_type(mark, 
                           new SharedBlock<State>(protocol, owner, addr_ + ptroff_t(offset), 
                            shadow_ + offset, acceleratorAddr_ + offset, blockSize, init)));
            size -= blockSize;
            offset += ptroff_t(blockSize);
        }
        TRACE(LOCAL, "Creating Shared Object @ %p : shadow @ %p : accelerator @ %p) ", addr_, shadow_, (void *) acceleratorAddr_);
    }
}


template<typename State>
SharedObject<State>::~SharedObject()
{
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->getBitmap();
    bitmap.unregisterRange(acceleratorAddr_, size_);
#endif

	// If the object creation failed, this address will be NULL
    if (acceleratorAddr_ != 0) owner_->unmap(addr_, size_);
    if (valid_) Memory::unshadow(shadow_, size_);
    if (addr_ != NULL) Memory::unmap(addr_, size_);
    TRACE(LOCAL, "Destroying Shared Object @ %p", addr_);
}

template<typename State>
inline accptr_t SharedObject<State>::acceleratorAddr(core::Mode &current, const hostptr_t addr) const
{
    accptr_t ret = accptr_t(0);
    lockRead();
    if(acceleratorAddr_ != 0) {
        ptroff_t offset = ptroff_t(addr - addr_);
        ret = acceleratorAddr_ + offset;
    }
    unlock();
    return ret;
}

template<typename State>
inline core::Mode &SharedObject<State>::owner(core::Mode &current, const hostptr_t addr) const
{
    lockRead();
    core::Mode &ret = *owner_;
    unlock();
    return ret;
}

template<typename State>
inline gmacError_t SharedObject<State>::addOwner(core::Mode &owner)
{
	return gmacErrorUnknown; // This kind of objects only accepts one owner
}

template<typename State>
inline
gmacError_t SharedObject<State>::removeOwner(core::Mode &owner)
{
    lockWrite();
    if(owner_ == &owner) {
        // Put myself in the orphan map
        owner.insertOrphan(*this);

        TRACE(LOCAL, "Shared Object @ %p is going orphan", addr_);
        if(acceleratorAddr_ != 0) {
            gmacError_t ret = coherenceOp(&Protocol::deleteBlock);
            ASSERTION(ret == gmacSuccess);
            ret = coherenceOp(&Protocol::unmapFromAccelerator);
            ASSERTION(ret == gmacSuccess);
            owner_->unmap(addr_, size_);
        }
        // Clean-up 
        BlockMap::iterator i;
        for(i = blocks_.begin(); i != blocks_.end(); i++) {
            i->second->release();
        }
        blocks_.clear();

        acceleratorAddr_ = accptr_t(0);
        owner_ = NULL;
    }
    unlock();
	return gmacSuccess;
}

template<typename State>
inline
gmacError_t SharedObject<State>::unmapFromAccelerator()
{
    lockWrite();
    // Remove blocks from the coherency domain
    gmacError_t ret = coherenceOp(&Protocol::unmapFromAccelerator);
    // Free accelerator memory
    CFATAL(owner_->unmap(addr_, size_) == gmacSuccess, "Error unmapping object from accelerator");
    unlock();
    return ret;
}


template<typename State>
inline gmacError_t SharedObject<State>::mapToAccelerator()
{
    lockWrite();
    gmacError_t ret;

    // Allocate accelerator memory in the new mode
    accptr_t newAcceleratorAddr = allocAcceleratorMemory(*owner_, addr_, size_);
    valid_ = (newAcceleratorAddr != 0);

    if (valid_) {
        acceleratorAddr_ = newAcceleratorAddr;
        // Recreate accelerator blocks
        repopulateBlocks(acceleratorAddr_, *owner_);
        // Add blocks to the coherency domain
        ret = coherenceOp(&Protocol::mapToAccelerator);
    }
    else {
        ret = gmacErrorMemoryAllocation;
    }


    unlock();
	return ret;
}

}}

#endif
