#ifndef GMAC_MEMORY_SHAREDBLOCK_IMPL_H_
#define GMAC_MEMORY_SHAREDBLOCK_IMPL_H_

#include "core/Mode.h"
#include "core/IOBuffer.h"

namespace __impl { namespace memory {

template<typename T>
inline SharedBlock<T>::SharedBlock(Protocol &protocol, core::Mode &owner, uint8_t *hostAddr,
								   uint8_t *shadowAddr, uint8_t *acceleratorAddr, size_t size, T init) :
	memory::StateBlock<T>(protocol, hostAddr, shadowAddr, size, init),
	owner_(owner),
	acceleratorAddr_(acceleratorAddr)
{}

template<typename T>
inline SharedBlock<T>::~SharedBlock()
{}

template<typename T>
inline core::Mode &SharedBlock<T>::owner() const
{
	return owner_;
}

template<typename T>
inline void *SharedBlock<T>::acceleratorAddr(const void *addr) const
{
	unsigned offset = unsigned((uint8_t *)addr - StateBlock<T>::addr_);
	return (void *)(acceleratorAddr_ + offset);
}

template<typename T>
inline gmacError_t SharedBlock<T>::toHost() const
{
	return owner_.copyToHost(StateBlock<T>::shadow_, acceleratorAddr_, StateBlock<T>::size_);
}

template<typename T>
inline gmacError_t SharedBlock<T>::toAccelerator() const
{
	return owner_.copyToAccelerator(acceleratorAddr_, StateBlock<T>::shadow_, StateBlock<T>::size_);
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyToHost(const void *src, size_t size, unsigned blockOffset) const
{
    ::memcpy(StateBlock<T>::shadow_ + blockOffset, src, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyToHost(core::IOBuffer &buffer, size_t size, 
											  unsigned bufferOffset, unsigned blockOffset) const
{
	::memcpy(StateBlock<T>::shadow_ + blockOffset, (uint8_t *)buffer.addr() + bufferOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyToAccelerator(const void *src, size_t size,  unsigned blockOffset) const
{
    return owner_.copyToAccelerator(acceleratorAddr_ + blockOffset, src, size);
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyToAccelerator(core::IOBuffer &buffer, size_t size, 
												unsigned bufferOffset, unsigned blockOffset) const
{
	return owner_.bufferToAccelerator(acceleratorAddr_ + blockOffset, buffer, size, bufferOffset);
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyFromHost(void *dst, size_t size, unsigned blockOffset) const
{
    ::memcpy(dst, StateBlock<T>::shadow_ + blockOffset, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyFromHost(core::IOBuffer &buffer, size_t size, 
												unsigned bufferOffset, unsigned blockOffset) const
{
	::memcpy((uint8_t *)buffer.addr() + bufferOffset, StateBlock<T>::shadow_ + blockOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyFromAccelerator(void *dst, size_t size, unsigned blockOffset) const
{
    return owner_.copyToHost(dst, acceleratorAddr_ + blockOffset, size);
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyFromAccelerator(core::IOBuffer &buffer, size_t size, 
												  unsigned bufferOffset, unsigned blockOffset) const
{
	return owner_.acceleratorToBuffer(buffer, acceleratorAddr_ + blockOffset, size, bufferOffset);
}

template<typename T>
inline gmacError_t SharedBlock<T>::hostMemset(int v, size_t size, unsigned blockOffset) const
{
    ::memset(StateBlock<T>::shadow_ + blockOffset, v, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::acceleratorMemset(int v, size_t size, unsigned blockOffset) const
{
    return owner_.memset(acceleratorAddr_ + blockOffset, v, size);
}

}}

#endif
