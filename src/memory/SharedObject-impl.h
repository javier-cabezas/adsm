#ifndef GMAC_MEMORY_SHAREDOBJECT_IMPL_H_
#define GMAC_MEMORY_SHAREDOBJECT_IMPL_H_

#include "memory/SharedBlock.h"

namespace __impl { namespace memory {

template<typename T>
inline SharedObject<T>::SharedObject(Protocol &protocol, core::Mode &owner, void *cpuAddr, size_t size, T init) :
    Object(cpuAddr, size),
	owner_(&owner)
{
	// Allocate memory (if necessary)
	if(addr_ == NULL)
		addr_ = (uint8_t *)Memory::map(NULL, size, GMAC_PROT_READWRITE);
    if(addr_ == NULL) return;

    // Create a shadow mapping for the host memory
    shadow_ = (uint8_t *)Memory::shadow(addr_, size_);

    // Allocate accelerator memory
    gmacError_t ret = 
		owner_->malloc((void **)&acceleratorAddr_, size, (unsigned)paramPageSize);
	if(ret == gmacSuccess) {
#ifdef USE_VM
        vm::Bitmap &bitmap = owner_->dirtyBitmap();
        bitmap.newRange(acceleratorAddr_, size_);
#endif
        valid_ = true;
    }

	// Populate the block-set
	uint8_t *mark = addr_;
	unsigned offset = 0;
	while(size > 0) {
		size_t blockSize = (size > paramPageSize) ? paramPageSize : size;
		mark += blockSize;
		blocks_.insert(BlockMap::value_type(mark, 
			new SharedBlock<T>(protocol, *owner_, addr_ + offset, shadow_ + offset, acceleratorAddr_ + offset, blockSize, init)));
		size -= blockSize;
		offset += unsigned(blockSize);
	}
    TRACE(LOCAL, "Creating Shared Object @ %p : shadow @ %p : accelerator @ %p) ", 
        addr_, shadow_, acceleratorAddr_);
}


template<typename T>
inline SharedObject<T>::~SharedObject()
{
	// If the object creation failed, this address will be NULL
    if(acceleratorAddr_ != NULL) owner_->free(acceleratorAddr_);
    Memory::unshadow(shadow_, size_);
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.removeRange(acceleratorAddr_, size_);
#endif
    TRACE(LOCAL, "Destroying Shared Object @ %p", addr_);
}

template<typename T>
inline void *SharedObject<T>::acceleratorAddr(const void *addr) const
{
    void *ret = NULL;
    lockRead();
    if(acceleratorAddr_ != NULL) {
        unsigned offset = unsigned((uint8_t *)addr - addr_);
        ret = acceleratorAddr_ + offset;
    }
    unlock();
    return ret;
}

template<typename T>
inline core::Mode &SharedObject<T>::owner(const void *addr) const
{
    lockRead();
    core::Mode &ret = *owner_;
    unlock();
    return ret;
}

template<typename T>
inline bool SharedObject<T>::addOwner(core::Mode &owner)
{
	return false; // This kind of objects only accepts one owner
}

template<typename T>
inline void SharedObject<T>::removeOwner(const core::Mode &owner)
{
    lockWrite();
    if(owner_ == &owner) {
        TRACE(LOCAL, "Shared Object @ %p is going orphan", addr_);
        if(acceleratorAddr_ != NULL) {
            coherenceOp(&Protocol::remove);
            owner_->free(acceleratorAddr_);
        }
        acceleratorAddr_ = NULL;
        owner_ = NULL;
        // Put myself in the orphan map
        Map::insertOrphan(*this);
    }
    unlock();
	return;
}

}}

#endif
