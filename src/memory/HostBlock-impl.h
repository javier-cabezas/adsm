#ifndef GMAC_MEMORY_SHAREDBLOCK_IMPL_H_
#define GMAC_MEMORY_SHAREDBLOCK_IMPL_H_

#include "core/IOBuffer.h"

namespace __impl { namespace memory {

inline HostBlock::HostBlock(void *hostAddr, size_t size)
    Block(hostAddr, size)
{}

template<typename T>
inline HostBlock::~HostBlock()
{}

template<typename T>
inline gmacError_t HostBlock::toHost()
{
	return gmacSuccess;
}

template<typename T>
inline gmacError_t HostBlock::toDevice()
{
	return gmacSuccess;
}

template<typename T>
inline gmacError_t HostBlock::copyToHost(const IOBuffer &buffer, size_t size, unsigned bufferOffset, unsigned blockOffset) const
{
	::memcpy(hostAdr_ + blockOffset, (uint8_t *)buffer.addr + bufferOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t HostBlock::copyToDevice(const IOBuffer &buffer, size_t size, unsigned bufferOffset, unsigned blockOffset) const
{
	return gmacSuccess;
}

template<typename T>
inline gmacError_t HostBlock::copyFromHost(IOBuffer &buffer, size_t size, unsigned bufferOffset, unsigned blockOffset) const
{
	::memcpy((uint8_t *)buffer.addr() + bufferOffset, hostAddr_ + blockOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t HostBlock::copyFromDevice(IOBuffer &buffer, size_t size, unsigned bufferOffset, unsigned blockOffset) const
{
	return gmacSuccess;
}


}}

#endif
