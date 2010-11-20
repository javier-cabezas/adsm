#ifndef GMAC_MEMORY_SHAREDOBJECT_IMPL_H_
#define GMAC_MEMORY_SHAREDOBJECT_IMPL_H_

#include "memory/SharedBlock.h"

namespace __impl { namespace memory {

template<typename T>
inline SharedObject<T>::SharedObject(Protocol &protocol, core::Mode &owner, void *cpuAddr, size_t size, T init) :
    Object(cpuAddr, size),
	owner_(&owner)
{
	if(addr_ == NULL) return;
    // Allocate accelerator memory
    gmacError_t ret = 
		owner_->malloc((void **)&deviceAddr_, size, (unsigned)paramPageSize);
	if(ret == gmacSuccess) valid_ = true;

	// Populate the block-set
	uint8_t *mark = addr_;
	unsigned offset = 0;
	while(size > 0) {
		size_t blockSize = (size > paramPageSize) ? paramPageSize : size;
		mark += blockSize;
		blocks_.insert(BlockMap::value_type(mark, 
			new SharedBlock<T>(protocol, *owner_, addr_ + offset, shadow_ + offset, deviceAddr_ + offset, blockSize, init)));
		size -= blockSize;
		offset += unsigned(blockSize);
	}
    TRACE(LOCAL, "Creating Shared Object @ %p : shadow @ %p : device @ %p) ", addr_, shadow_, deviceAddr_);
}


template<typename T>
inline SharedObject<T>::~SharedObject()
{
	// If the object creation failed, this address will be NULL
    if(deviceAddr_ != NULL) owner_->free(deviceAddr_);	
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.removeRange(devAddr, StateObject<T>::size_);
#endif
    TRACE(LOCAL, "Destroying Shared Object @ %p", addr_);
}

template<typename T>
inline void *SharedObject<T>::deviceAddr(const void *addr) const
{
    void *ret = NULL;
    lockRead();
    if(deviceAddr_ != NULL) {
        unsigned offset = unsigned((uint8_t *)addr - addr_);
        ret = deviceAddr_ + offset;
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
        if(deviceAddr_ != NULL) {
            coherenceOp(&Protocol::remove);
            owner_->free(deviceAddr_);
        }
        deviceAddr_ = NULL;
        owner_ = NULL;
        // Put myself in the orphan map
        Map::insertOrphan(*this);
    }
    unlock();
	return;
}

}}

#endif
