#ifndef GMAC_MEMORY_BLOCK_IMPL_H_
#define GMAC_MEMORY_BLOCK_IMPL_H_

#include "memory/Memory.h"

namespace __impl { namespace memory { 

inline Block::Block(Protocol &protocol, uint8_t *addr, uint8_t *shadow, size_t size) :
	gmac::util::Lock("Block"),
	protocol_(protocol),
	size_(size),
	addr_(addr),
    shadow_(shadow)
{ }

inline Block::~Block()
{ }

inline uint8_t *Block::addr() const
{
	return addr_;
}

inline size_t Block::size() const
{
    return size_;
}

inline gmacError_t Block::signalRead()
{
	lock();
	gmacError_t ret = protocol_.signalRead(*this);
	unlock();
	return ret;
}

inline gmacError_t Block::signalWrite()
{
	lock();
	gmacError_t ret = protocol_.signalWrite(*this);
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

}}

#endif
