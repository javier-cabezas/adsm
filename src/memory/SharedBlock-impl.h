#ifndef GMAC_MEMORY_SHAREDBLOCK_IMPL_H_
#define GMAC_MEMORY_SHAREDBLOCK_IMPL_H_

#include "core/Mode.h"
#include "core/IOBuffer.h"

namespace __impl { namespace memory {

template<typename State>
inline
SharedBlock<State>::SharedBlock(Protocol &protocol, core::Mode &owner, hostptr_t hostAddr,
                                                           hostptr_t shadowAddr, accptr_t acceleratorAddr, size_t size, typename State::ProtocolState init) :
        memory::StateBlock<State>(protocol, hostAddr, shadowAddr, size, init),
        owner_(owner),
        acceleratorAddr_(acceleratorAddr)
{}

template<typename State>
inline
SharedBlock<State>::~SharedBlock()
{}

template<typename State>
inline core::Mode &
SharedBlock<State>::owner(core::Mode &current) const
{
        return owner_;
}

template<typename State>
inline accptr_t
SharedBlock<State>::acceleratorAddr(core::Mode &current, const hostptr_t addr) const
{
        ptroff_t offset = ptroff_t(addr - StateBlock<State>::addr_);
    accptr_t ret = acceleratorAddr_ + offset;
        return ret;
}

template<typename State>
inline accptr_t
SharedBlock<State>::acceleratorAddr(core::Mode &current) const
{
    return acceleratorAddr_;
}

template<typename State>
inline gmacError_t
SharedBlock<State>::toHost(unsigned blockOffset, size_t count)
{
    gmacError_t ret = gmacSuccess;
    ret = owner_.copyToHost(StateBlock<State>::shadow_ + blockOffset, acceleratorAddr_ + blockOffset, count);
    return ret;
}

template<typename State>
inline gmacError_t
SharedBlock<State>::toAccelerator(unsigned blockOffset, size_t count)
{
    gmacError_t ret = gmacSuccess;
    ret = owner_.copyToAccelerator(acceleratorAddr_ + blockOffset, StateBlock<State>::shadow_ + blockOffset, count);
    return ret;
}


template<typename State>
inline gmacError_t
SharedBlock<State>::copyFromBuffer(size_t blockOff, core::IOBuffer &buffer,
                                   size_t bufferOff, size_t size, typename StateBlock<State>::Destination dst) const
{
    gmacError_t ret = gmacSuccess;
    switch (dst) {
    case StateBlock<State>::HOST:
        ::memcpy(StateBlock<State>::shadow_ + blockOff, buffer.addr() + bufferOff, size); return gmacSuccess;
        break;
    case StateBlock<State>::ACCELERATOR:
        ret = owner_.bufferToAccelerator(acceleratorAddr_ + ptroff_t(blockOff), buffer, size, bufferOff);
        break;
    }

    return ret;
}

template<typename State>
inline gmacError_t
SharedBlock<State>::copyToBuffer(core::IOBuffer &buffer, size_t bufferOff, size_t blockOff,
                                 size_t size, typename StateBlock<State>::Source src) const
{
    gmacError_t ret = gmacSuccess;
    switch (src) {
    case StateBlock<State>::HOST:
        ::memcpy(buffer.addr() + bufferOff, StateBlock<State>::shadow_ + blockOff, size);
        break;
    case StateBlock<State>::ACCELERATOR:
        ret = owner_.acceleratorToBuffer(buffer, acceleratorAddr_ + ptroff_t(blockOff), size, bufferOff);
        break;
    }

    return ret;
}

template<typename State>
gmacError_t
SharedBlock<State>::copyFromBlock(size_t dstOff, StateBlock<State> &srcBlock,
                                  size_t srcOff, size_t size,
                                  typename StateBlock<State>::Source src,
                                  typename StateBlock<State>::Destination dst) const
{
    gmacError_t ret = gmacSuccess;
    if (src == StateBlock<State>::ACCELERATOR &&
        dst == StateBlock<State>::ACCELERATOR) {
        TRACE(LOCAL, "A -> A");
        ret = owner_.copyAccelerator(acceleratorAddr_ + dstOff, srcBlock.acceleratorAddr(srcBlock.owner(owner_)) + srcOff, size);
        TRACE(LOCAL, "RESULT: %d", ret);
    } else if (src == StateBlock<State>::HOST &&
               dst == StateBlock<State>::HOST) {
        TRACE(LOCAL, "H -> H");
        ::memcpy(Block::shadow_ + dstOff, srcBlock.getShadow() + srcOff, size);
    } else if (src == StateBlock<State>::HOST &&
               dst == StateBlock<State>::ACCELERATOR) {
        TRACE(LOCAL, "H -> A");
        ret = owner_.copyToAccelerator(acceleratorAddr_ + dstOff,
                                       srcBlock.getShadow() + srcOff, size);
        TRACE(LOCAL, "RESULT: %d", ret);
    } else if (src == StateBlock<State>::ACCELERATOR &&
               dst == StateBlock<State>::HOST) {
        TRACE(LOCAL, "A -> H");
        ret = owner_.copyToHost(Block::shadow_ + dstOff, srcBlock.acceleratorAddr(srcBlock.owner(owner_)) + srcOff, size);
        TRACE(LOCAL, "RESULT: %d", ret);
    }

    return ret;
}

#if 0
template<typename State>
inline gmacError_t
SharedBlock<State>::copyToHost(const hostptr_t src, size_t size, size_t blockOffset) const
{
    ::memcpy(StateBlock<State>::shadow_ + blockOffset, src, size);
    return gmacSuccess;
}

template<typename State>
inline gmacError_t
SharedBlock<State>::copyToAccelerator(const hostptr_t src, size_t size,  size_t blockOffset) const
{
    return owner_.copyToAccelerator(acceleratorAddr_ + ptroff_t(blockOffset), src, size);
}

template<typename State>
inline gmacError_t
SharedBlock<State>::copyFromHost(hostptr_t dst, size_t size, size_t blockOffset) const
{
    ::memcpy(dst, StateBlock<State>::shadow_ + blockOffset, size);
    return gmacSuccess;
}

template<typename State>
inline gmacError_t
SharedBlock<State>::copyFromAccelerator(hostptr_t dst, size_t size, size_t blockOffset) const
{
    return owner_.copyToHost(dst, acceleratorAddr_ + ptroff_t(blockOffset), size);
}
#endif

template<typename State>
inline gmacError_t
SharedBlock<State>::hostMemset(int v, size_t size, size_t blockOffset) const
{
    ::memset(StateBlock<State>::shadow_ + blockOffset, v, size);
    return gmacSuccess;
}

template<typename State>
inline gmacError_t
SharedBlock<State>::acceleratorMemset(int v, size_t size, size_t blockOffset) const
{
    return owner_.memset(acceleratorAddr_ + ptroff_t(blockOffset), v, size);
}

}}

#endif
