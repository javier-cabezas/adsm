#ifndef GMAC_MEMORY_SHAREDOBJECT_IMPL_H_
#define GMAC_MEMORY_SHAREDOBJECT_IMPL_H_

#include "memory/SharedBlock.h"

namespace __impl { namespace memory {

template<typename T>
inline SharedObject<T>::SharedObject(Protocol &protocol, core::Mode &owner, void *cpuAddr, size_t size, T init) :
    Object(cpuAddr, size),
	owner_(owner)
{
	if(addr_ == NULL) return;
    // Allocate accelerator memory
    gmacError_t ret = 
		owner_.malloc((void **)&deviceAddr_, size, (unsigned)paramPageSize);
	if(ret == gmacSuccess) valid_ = true;

	// Populate the block-set
	uint8_t *mark = addr_;
	unsigned offset = 0;
	while(size > 0) {
		size_t blockSize = (size > paramPageSize) ? paramPageSize : size;
		mark += blockSize;
		blocks_.insert(BlockMap::value_type(mark, 
			new SharedBlock<T>(protocol, owner_, addr_ + offset, deviceAddr_ + offset, blockSize, init)));
		size -= blockSize;
		offset += unsigned(blockSize);
	}
}


template<typename T>
inline SharedObject<T>::~SharedObject()
{
	// If the object creation failed, this address will be NULL
    if(deviceAddr_ != NULL) owner_.free(deviceAddr_);	
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.removeRange(devAddr, StateObject<T>::size_);
#endif
}

template<typename T>
inline void SharedObject<T>::addOwner(core::Mode &owner)
{
	return;
}

template<typename T>
inline void SharedObject<T>::removeOwner(core::Mode &owner)
{
	return;
}

}}

#endif
