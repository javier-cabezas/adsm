#ifndef GMAC_MEMORY_BLOCK_IMPL_H_
#define GMAC_MEMORY_BLOCK_IMPL_H_

#include "memory/Memory.h"
#ifdef USE_VM
#include "Bitmap.h"
#include "core/Mode.h"
#endif

namespace __impl { namespace memory { 

inline Block::Block(Protocol &protocol, uint8_t *addr, uint8_t *shadow, size_t size) :
	gmac::util::Lock("Block"),
	protocol_(protocol),
	size_(size),
	addr_(addr),
    shadow_(shadow)
{
#ifdef USE_VM
    resetBitmapStats();
    subBlockSize_ = paramPageSize/paramBitmapChunksPerPage; 
#endif
}

inline Block::~Block()
{ }

#ifdef USE_VM
inline void
Block::resetBitmapStats()
{
    faults_           = 0;
    sequentialFaults_ = 0;
}

inline void
Block::updateBitmapStats(const void *addr, bool write)
{
    core::Mode &mode = owner();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    unsigned currentSubBlock = bitmap.getSubBlock(acceleratorAddr(addr));
#if 0
    if (write) {
        bitmap.set(acceleratorAddr(addr));
    }
#endif

    if (faults_ > 0) {
        if (currentSubBlock == lastSubBlock_ + 1) {
            sequentialFaults_++;
        } else {
            sequentialFaults_ = 1;
        }
    } else {
        sequentialFaults_ = 1;
    }
    lastSubBlock_ = currentSubBlock;
    faults_++;
}

inline bool
Block::isSequentialAccess() const
{
    return sequentialFaults_ >= 2;
}

inline unsigned
Block::getFaults() const
{
    return faults_;
}

inline unsigned
Block::getSequentialFaults() const
{
    return sequentialFaults_;
}

inline void *
Block::getSubBlockAddr(const void *addr) const
{
    core::Mode &mode = owner();
    const vm::Bitmap &bitmap = mode.dirtyBitmap();
    unsigned subBlock = bitmap.getSubBlock(acceleratorAddr(addr));
    return addr_ + (subBlock * subBlockSize_);
}

inline size_t
Block::getSubBlockSize() const
{
    return subBlockSize_;
}

inline unsigned
Block::getSubBlocks() const
{
    return size_/subBlockSize_;
}

inline bool
Block::isSubBlockPresent(const void *addr) const
{
    core::Mode &mode = owner();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    return !bitmap.check(acceleratorAddr(addr));
}

inline void 
Block::setSubBlockPresent(const void *addr)
{
    core::Mode &mode = owner();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    bitmap.set(acceleratorAddr(addr));
}

inline void 
Block::setBlockPresent()
{
    core::Mode &mode = owner();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    bitmap.setBlock(acceleratorAddr(addr_));
}

#endif

inline uint8_t *Block::addr() const
{
	return addr_;
}

inline size_t Block::size() const
{
    return size_;
}

inline gmacError_t Block::signalRead(void *addr)
{
    TRACE(LOCAL,"SIGNAL READ on block %p: addr %p", addr_, addr);
	lock();
	gmacError_t ret = protocol_.signalRead(*this, addr);
	unlock();
	return ret;
}

inline gmacError_t Block::signalWrite(void *addr)
{
    TRACE(LOCAL,"SIGNAL WRITE on block %p: addr %p", addr_, addr);
	lock();
	gmacError_t ret = protocol_.signalWrite(*this, addr);
#ifdef USE_VM
    updateBitmapStats(addr, true);
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
		unsigned bufferOffset, unsigned blockOffset)
{
	lock();
	gmacError_t ret =(protocol_.*op)(*this, buffer, size, bufferOffset, blockOffset);
	unlock();
	return ret;
}

inline gmacError_t Block::memset(int v, size_t size, unsigned blockOffset) const
{
    lock();
    gmacError_t ret = protocol_.memset(*this, v, size, blockOffset);
    unlock();
    return ret;
}

}}

#endif
