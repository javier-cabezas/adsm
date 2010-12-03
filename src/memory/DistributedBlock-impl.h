#ifndef GMAC_MEMORY_DISTRIBUTEDBLOCK_IMPL_H_
#define GMAC_MEMORY_DISTRIBUTEDBLOCK_IMPL_H_

#include "core/Mode.h"
#include "core/IOBuffer.h"

namespace __impl { namespace memory {

template<typename T>
inline DistributedBlock<T>::DistributedBlock(Protocol &protocol, core::Mode &owner, uint8_t *hostAddr,
											 uint8_t *shadowAddr, uint8_t *acceleratorAddr, size_t size, T init) :
    StateBlock<T>(protocol, hostAddr, shadowAddr, size, init)
{
	acceleratorAddr_.insert(AcceleratorMap::value_type(&owner, acceleratorAddr));
}

template<typename T>
inline DistributedBlock<T>::~DistributedBlock()
{}

template<typename T>
inline void DistributedBlock<T>::addOwner(core::Mode &mode, uint8_t *value)
{
	StateBlock<T>::lock();
	std::pair<AcceleratorMap::iterator, bool> pair = 
        acceleratorAddr_.insert(AcceleratorMap::value_type(&mode, value));
    ASSERTION(pair.second == true);
    if(StateBlock<T>::protocol_.needUpdate(*this) == true) {
        gmacError_t ret = mode.copyToAccelerator(value, StateBlock<T>::shadow_, StateBlock<T>::size_);
        ASSERTION(ret == gmacSuccess);
    }
    StateBlock<T>::unlock();
}

template<typename T>
inline void DistributedBlock<T>::removeOwner(core::Mode &mode)
{
	StateBlock<T>::lock();
	acceleratorAddr_.erase(&mode);
	StateBlock<T>::unlock();
}

template<typename T>
inline core::Mode &DistributedBlock<T>::owner() const
{
	return core::Mode::current();
}

template<typename T>
inline void *DistributedBlock<T>::acceleratorAddr(const void *addr) const
{
	void *ret = NULL;

	StateBlock<T>::lock();
	AcceleratorMap::const_iterator i;
	i = acceleratorAddr_.find(&core::Mode::current());
	if(i != acceleratorAddr_.end()) {
		ret = (void *)(i->second + ((uint8_t *)addr - StateBlock<T>::addr_));
	}
	StateBlock<T>::unlock();
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::toHost() const
{
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::toAccelerator()
{
	gmacError_t ret = gmacSuccess;
	AcceleratorMap::const_iterator i;
	for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
		ret = i->first->copyToAccelerator(i->second, StateBlock<T>::shadow_, StateBlock<T>::size_);
		if(ret != gmacSuccess) break;
	}
#ifdef USE_VM
    Block::resetBitmapStats();
#endif
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyToHost(const void *src, size_t size, unsigned blockOffset) const
{
    ::memcpy(StateBlock<T>::shadow_ + blockOffset, src, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyToHost(core::IOBuffer &buffer, size_t size, 
												unsigned bufferOffset, unsigned blockOffset) const
{
	::memcpy(StateBlock<T>::shadow_ + blockOffset, (uint8_t *)buffer.addr() + bufferOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyToAccelerator(const void *src, size_t size,  unsigned blockOffset) const
{
    gmacError_t ret = gmacSuccess;
	AcceleratorMap::const_iterator i;
	for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
		ret = i->first->copyToAccelerator(i->second + blockOffset, src, size);
		if(ret != gmacSuccess) return ret;
	}
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyToAccelerator(core::IOBuffer &buffer, size_t size, 
												  unsigned bufferOffset, unsigned blockOffset) const
{
	gmacError_t ret = gmacSuccess;
	AcceleratorMap::const_iterator i;
	for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
		ret = i->first->bufferToAccelerator(i->second + blockOffset, buffer, size, bufferOffset);
		if(ret != gmacSuccess) return ret;
	}
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyFromHost(void *dst, size_t size, unsigned blockOffset) const
{
    ::memcpy(dst, StateBlock<T>::shadow_ + blockOffset, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyFromHost(core::IOBuffer &buffer, size_t size, 
												  unsigned bufferOffset, unsigned blockOffset) const
{
	::memcpy((uint8_t *)buffer.addr() + bufferOffset, StateBlock<T>::shadow_ + blockOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyFromAccelerator(void *dst, size_t size, unsigned blockOffset) const
{
    ::memcpy(dst, StateBlock<T>::shadow_ + blockOffset, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyFromAccelerator(core::IOBuffer &buffer, size_t size, 
													unsigned bufferOffset, unsigned blockOffset) const
{
	::memcpy((uint8_t *)buffer.addr() + bufferOffset, StateBlock<T>::shadow_ + blockOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::hostMemset(int v, size_t size, unsigned blockOffset) const
{
    ::memset(StateBlock<T>::shadow_ + blockOffset, v, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::acceleratorMemset(int v, size_t size, unsigned blockOffset) const
{
    gmacError_t ret = gmacSuccess;
	AcceleratorMap::const_iterator i;
	for(i = acceleratorAddr_.begin(); i != acceleratorAddr_.end(); i++) {
        ret = i->first->memset(i->second + blockOffset, v, size);
		if(ret != gmacSuccess) break;
	}
	return ret;
}

}}

#endif
