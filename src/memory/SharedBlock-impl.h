#ifndef GMAC_MEMORY_SHAREDBLOCK_IMPL_H_
#define GMAC_MEMORY_SHAREDBLOCK_IMPL_H_

#include "core/Mode.h"
#include "core/IOBuffer.h"

namespace __impl { namespace memory {

template<typename State>
inline SharedBlock<State>::SharedBlock(Protocol &protocol, core::Mode &owner, hostptr_t hostAddr,
							   hostptr_t shadowAddr, accptr_t acceleratorAddr, size_t size, typename State::ProtocolState init) :
	memory::StateBlock<State>(protocol, hostAddr, shadowAddr, size, init),
	owner_(owner),
	acceleratorAddr_(acceleratorAddr)
{}

template<typename State>
inline SharedBlock<State>::~SharedBlock()
{}

template<typename State>
inline core::Mode &SharedBlock<State>::owner(core::Mode &current) const
{
	return owner_;
}

template<typename State>
inline accptr_t SharedBlock<State>::acceleratorAddr(core::Mode &current, const hostptr_t addr) const
{
	ptroff_t offset = ptroff_t(addr - StateBlock<State>::addr_);
    accptr_t ret = acceleratorAddr_ + offset;
	return ret;
}

template<typename State>
inline accptr_t SharedBlock<State>::acceleratorAddr(core::Mode &current) const
{
    return acceleratorAddr_;
}

template<typename State>
inline gmacError_t SharedBlock<State>::toHost(unsigned blockOffset, size_t count)
{
    gmacError_t ret = gmacSuccess;
    ret = owner_.copyToHost(StateBlock<State>::shadow_ + blockOffset, acceleratorAddr_ + blockOffset, count);
    return ret;
}

template<typename State>
inline gmacError_t SharedBlock<State>::toAccelerator(unsigned blockOffset, size_t count)
{
    gmacError_t ret = gmacSuccess;
    ret = owner_.copyToAccelerator(acceleratorAddr_ + blockOffset, StateBlock<State>::shadow_ + blockOffset, count);
    return ret;
}

template<typename State>
inline gmacError_t SharedBlock<State>::copyToHost(const hostptr_t src, size_t size, size_t blockOffset) const
{
    ::memcpy(StateBlock<State>::shadow_ + blockOffset, src, size);
    return gmacSuccess;
}

template<typename State>
inline gmacError_t SharedBlock<State>::copyToHost(core::IOBuffer &buffer, size_t size, 
											  size_t bufferOffset, size_t blockOffset) const
{
	::memcpy(StateBlock<State>::shadow_ + blockOffset, buffer.addr() + bufferOffset, size);
	return gmacSuccess;
}

template<typename State>
inline gmacError_t SharedBlock<State>::copyToAccelerator(const hostptr_t src, size_t size,  size_t blockOffset) const
{
    return owner_.copyToAccelerator(acceleratorAddr_ + ptroff_t(blockOffset), src, size);
}

template<typename State>
inline gmacError_t SharedBlock<State>::copyToAccelerator(core::IOBuffer &buffer, size_t size, 
												size_t bufferOffset, size_t blockOffset) const
{
	return owner_.bufferToAccelerator(acceleratorAddr_ + ptroff_t(blockOffset), 
        buffer, size, bufferOffset);
}

template<typename State>
inline gmacError_t SharedBlock<State>::copyFromHost(hostptr_t dst, size_t size, size_t blockOffset) const
{
    ::memcpy(dst, StateBlock<State>::shadow_ + blockOffset, size);
    return gmacSuccess;
}

template<typename State>
inline gmacError_t SharedBlock<State>::copyFromHost(core::IOBuffer &buffer, size_t size, 
												size_t bufferOffset, size_t blockOffset) const
{
	::memcpy(buffer.addr() + bufferOffset, StateBlock<State>::shadow_ + blockOffset, size);
	return gmacSuccess;
}

template<typename State>
inline gmacError_t SharedBlock<State>::copyFromAccelerator(hostptr_t dst, size_t size, size_t blockOffset) const
{
    return owner_.copyToHost(dst, acceleratorAddr_ + ptroff_t(blockOffset), size);
}

template<typename State>
inline gmacError_t SharedBlock<State>::copyFromAccelerator(core::IOBuffer &buffer, size_t size, 
												  size_t bufferOffset, size_t blockOffset) const
{
	return owner_.acceleratorToBuffer(buffer, acceleratorAddr_ + ptroff_t(blockOffset), 
        size, bufferOffset);
}

template<typename State>
inline gmacError_t SharedBlock<State>::hostMemset(int v, size_t size, size_t blockOffset) const
{
    ::memset(StateBlock<State>::shadow_ + blockOffset, v, size);
    return gmacSuccess;
}

template<typename State>
inline gmacError_t SharedBlock<State>::acceleratorMemset(int v, size_t size, size_t blockOffset) const
{
    return owner_.memset(acceleratorAddr_ + ptroff_t(blockOffset), v, size);
}

}}

#endif
