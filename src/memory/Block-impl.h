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
}

inline Block::~Block()
{
}

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
	unlock();
	return ret;
}

inline gmacError_t Block::coherenceOp(gmacError_t (Protocol::*f)(Block &))
{
	lock();
	gmacError_t ret = (protocol_.*f)(*this);
	unlock();
	return ret;
}

inline gmacError_t Block::memoryOp(Protocol::MemoryOp op,
                                   core::IOBuffer &buffer, size_t size, size_t bufferOffset, size_t blockOffset)
{
	lock();
	gmacError_t ret =(protocol_.*op)(*this, buffer, size, bufferOffset, blockOffset);
	unlock();
	return ret;
}

inline gmacError_t Block::memset(int v, size_t size, size_t blockOffset)
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

inline gmacError_t Block::dump(std::ostream &param, protocol::common::Statistic stat)
{
    lock();
    gmacError_t ret = protocol_.dump(*this, param, stat);
    unlock();
    return ret;
}

}}

#endif
