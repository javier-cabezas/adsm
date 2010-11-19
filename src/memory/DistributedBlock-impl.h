#ifndef GMAC_MEMORY_DISTRIBUTEDBLOCK_IMPL_H_
#define GMAC_MEMORY_DISTRIBUTEDBLOCK_IMPL_H_

#include "core/Mode.h"
#include "core/IOBuffer.h"

namespace __impl { namespace memory {

template<typename T>
inline DistributedBlock<T>::DistributedBlock(Protocol &protocol, core::Mode &owner, 
											 void *hostAddr, void *deviceAddr, 
											 size_t size, T init) :
    StateBlock<T>(protocol, (uint8_t *)hostAddr, size, init)
{
	deviceAddr_.insert(DeviceMap::value_type(&owner, (uint8_t *)deviceAddr));
}

template<typename T>
inline DistributedBlock<T>::~DistributedBlock()
{}

template<typename T>
inline void DistributedBlock<T>::addOwner(core::Mode &mode, uint8_t *value)
{
	lock();
	deviceAddr_.insert(DeviceMap::value_type(&mode, value));
	unlock();
}

template<typename T>
inline void DistributedBlock<T>::removeOwner(core::Mode &mode)
{
	lock();
	deviceAddr_.erase(&mode);
	unlock();
}

template<typename T>
inline core::Mode &DistributedBlock<T>::owner() const
{
	return core::Mode::current();
}

template<typename T>
inline void *DistributedBlock<T>::deviceAddr(const void *addr) const
{
	void *ret = NULL;
	lock();
	DeviceMap::const_iterator i;
	i = deviceAddr_.find(&core::Mode::current());
	if(i != deviceAddr_.end()) {
		ret = (void *)(i->second + ((uint8_t *)addr - addr_));
	}
	unlock();
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::toHost() const
{
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::toDevice() const
{
	gmacError_t ret = gmacSuccess;
	DeviceMap::const_iterator i;
	lock();
	for(i = deviceAddr_.begin(); i != deviceAddr_.end(); i++) {
		ret = i->first->copyToAccelerator(i->second, shadow_, size_);
		if(ret != gmacSuccess) break;
	}
	unlock();
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyToHost(core::IOBuffer &buffer, size_t size, 
												unsigned bufferOffset, unsigned blockOffset) const
{
	::memcpy(shadow_ + blockOffset, (uint8_t *)buffer.addr() + bufferOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyToDevice(core::IOBuffer &buffer, size_t size, 
												  unsigned bufferOffset, unsigned blockOffset) const
{
	gmacError_t ret = gmacSuccess;
	DeviceMap::const_iterator i;
	lock();
	for(i = deviceAddr_.begin(); i != deviceAddr_.end(); i++) {
		ret = i->first->bufferToAccelerator(i->second + blockOffset, buffer, size, bufferOffset);
		if(ret != gmacSuccess) return ret;
	}
	unlock();
	return ret;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyFromHost(core::IOBuffer &buffer, size_t size, 
												  unsigned bufferOffset, unsigned blockOffset) const
{
	::memcpy((uint8_t *)buffer.addr() + bufferOffset, shadow_ + blockOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t DistributedBlock<T>::copyFromDevice(core::IOBuffer &buffer, size_t size, 
													unsigned bufferOffset, unsigned blockOffset) const
{
	::memcpy((uint8_t *)buffer.addr() + bufferOffset, shadow_ + blockOffset, size);
	return gmacSuccess;
}


}}

#endif
