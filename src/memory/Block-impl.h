#ifndef GMAC_MEMORY_BLOCK_IMPL_H_
#define GMAC_MEMORY_BLOCK_IMPL_H_

#include "memory/Memory.h"
#ifdef USE_VM
#include "vm/Bitmap.h"
#include "core/Mode.h"
#endif

namespace __impl { namespace memory { 

inline Block::Block(Protocol &protocol, hostptr_t addr, hostptr_t shadow, size_t size) :
	gmac::util::Lock("Block"),
	protocol_(protocol),
	size_(size),
	addr_(addr),
    shadow_(shadow)
{
#if defined(USE_VM) || defined(USE_SUBBLOCK_TRACKING)
    resetSubBlockStats();
#if defined(USE_SUBBLOCK_TRACKING)
    for (size_t s = 0; s < size; s += SubBlockSize_) {
        subBlockState_.push_back(0);
    }
#endif
#endif
}

inline Block::~Block()
{ }

#if defined(USE_VM) || defined(USE_SUBBLOCK_TRACKING)
inline void
Block::resetSubBlockStats()
{
    faults_        = 0;
    stridedFaults_ = 0;
}

inline void
Block::updateSubBlockStats(const hostptr_t addr, bool write)
{
    core::Mode &mode = owner();
    long_t currentSubBlock = GetSubBlock(addr);
#if 0
    if (write) {
        bitmap.set(acceleratorAddr(addr));
    }
#endif

    if (faults_ == 1) {
        stride_ = addr - lastAddr_;
        stridedFaults_  = 2;
    } else if (faults_ > 1) {
        if (addr == lastAddr_ + stride_) {
            stridedFaults_++;
        } else {
            stridedFaults_ = 1;
        }
    } else {
        stridedFaults_ = 1;
    }
    lastAddr_ = addr;
    faults_++;
}

inline bool
Block::isStridedAccess() const
{
    /// \todo Do not hardcode the threshold value
    return stridedFaults_ > 2;
}

inline unsigned
Block::getFaults() const
{
    return faults_;
}

inline unsigned
Block::getStridedFaults() const
{
    return stridedFaults_;
}

inline hostptr_t
Block::getSubBlockAddr(const hostptr_t addr) const
{
    return GetSubBlockAddr(addr_, addr);
}

inline size_t
Block::getSubBlockSize() const
{
    return size_ < SubBlockSize_? size_: SubBlockSize_;
}

inline unsigned
Block::getSubBlocks() const
{
    unsigned subBlocks = size_/SubBlockSize_;
    if (size_ % SubBlockSize_ != 0) subBlocks++;
    return subBlocks;
}

inline void 
Block::setSubBlockDirty(const hostptr_t addr)
{
#ifdef USE_VM
    core::Mode &mode = owner();
    vm::BitmapShared &bitmap = mode.acceleratorDirtyBitmap();
    bitmap.setEntry(acceleratorAddr(addr), vm::BITMAP_SET_HOST);
#else // USE_SUBBLOCK_TRACKING
    subBlockState_[GetSubBlockIndex(addr_, addr)] = 1;
#endif
}

inline void 
Block::setBlockDirty()
{
#ifdef USE_VM
    core::Mode &mode = owner();
    vm::BitmapShared &bitmap = mode.acceleratorDirtyBitmap();
    bitmap.setEntryRange(acceleratorAddr(addr_), size_, vm::BITMAP_SET_HOST);
#else // USE_SUBBLOCK_TRACKING
    SubBlockVector::iterator it;
    for (it = subBlockState_.begin(); it != subBlockState_.end(); it++) {
        *it = 1;
    }
#endif
}

#endif

inline hostptr_t Block::addr() const
{
	return addr_;
}

inline size_t Block::size() const
{
    return size_;
}

inline gmacError_t Block::signalRead(hostptr_t addr)
{
    TRACE(LOCAL,"SIGNAL READ on block %p: addr %p", addr_, addr);
	lock();
	gmacError_t ret = protocol_.signalRead(*this, addr);
	unlock();
	return ret;
}

inline gmacError_t Block::signalWrite(hostptr_t addr)
{
    TRACE(LOCAL,"SIGNAL WRITE on block %p: addr %p", addr_, addr);
	lock();
	gmacError_t ret = protocol_.signalWrite(*this, addr);
#if defined(USE_VM) || defined(USE_SUBBLOCK_TRACKING)
    updateSubBlockStats(addr, true);
#endif
	unlock();
	return ret;
}

inline gmacError_t Block::coherenceOp(Protocol::CoherenceOp op)
{
	lock();
	gmacError_t ret = (protocol_.*op)(*this);
	unlock();
	return ret;
}

inline gmacError_t Block::memoryOp(Protocol::MemoryOp op, core::IOBuffer &buffer, size_t size, 
		size_t bufferOffset, size_t blockOffset)
{
	lock();
	gmacError_t ret =(protocol_.*op)(*this, buffer, size, bufferOffset, blockOffset);
	unlock();
	return ret;
}

inline gmacError_t Block::memset(int v, size_t size, size_t blockOffset) const
{
    lock();
    gmacError_t ret = protocol_.memset(*this, v, size, blockOffset);
    unlock();
    return ret;
}

inline
Protocol &Block::getProtocol()
{
    return protocol_;
}

}}

#endif
