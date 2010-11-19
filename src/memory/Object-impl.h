#ifndef GMAC_MEMORY_OBJECT_IMPL_H_
#define GMAC_MEMORY_OBJECT_IMPL_H_

#include "Protocol.h"
#include "Block.h"

#include "util/Logger.h"

namespace __impl { namespace memory {

inline Object::Object(void *addr, size_t size) :
	gmac::util::RWLock("Object"), 
	addr_((uint8_t *)addr),
	size_(size),
	valid_(false)
{
	// Allocate memory (if necessary)
	if(addr_ == NULL)
		addr_ = (uint8_t *)Memory::map(NULL, size, GMAC_PROT_READWRITE);
    shadow_ = (uint8_t *)Memory::shadow(addr_, size_);
}

inline Object::~Object()
{
    Memory::unshadow(shadow_, size_);
}

inline uint8_t *Object::addr() const
{
    return (uint8_t *) addr_;
}

inline uint8_t *Object::end() const
{
    return addr() + size_;
}

inline size_t Object::size() const
{
    return size_;
}

inline bool Object::valid() const
{
	return valid_;
}

inline gmacError_t Object::acquire() const
{
	return coherenceOp(&Protocol::acquire);
}

inline gmacError_t Object::toHost() const
{
	return coherenceOp(&Protocol::toHost);
}

inline gmacError_t Object::toDevice() const
{
	return coherenceOp(&Protocol::toDevice);
}

inline gmacError_t Object::signalRead(void *addr) const
{
	gmacError_t ret = gmacSuccess;
	lockRead();
	BlockMap::const_iterator i = blocks_.upper_bound((uint8_t *)addr);
	if(i == blocks_.end()) ret = gmacErrorInvalidValue;
	else if(i->second->addr() > addr) ret = gmacErrorInvalidValue;
	else ret = i->second->signalRead();
	unlock();
	return ret;
}

inline gmacError_t Object::signalWrite(void *addr) const
{
	gmacError_t ret = gmacSuccess;
	lockRead();
	BlockMap::const_iterator i = blocks_.upper_bound((uint8_t *)addr);
	if(i == blocks_.end()) ret = gmacErrorInvalidValue;
	else if(i->second->addr() > addr) ret = gmacErrorInvalidValue;
	else ret = i->second->signalWrite();
	unlock();
	return ret;
}

inline gmacError_t Object::copyToBuffer(core::IOBuffer &buffer, size_t size, 
									  unsigned bufferOffset, unsigned objectOffset) const
{
	return memoryOp(&Protocol::copyToBuffer, buffer, size, bufferOffset, objectOffset);
}

inline gmacError_t Object::copyFromBuffer(core::IOBuffer &buffer, size_t size, 
										unsigned bufferOffset, unsigned objectOffset) const
{
	return memoryOp(&Protocol::copyFromBuffer, buffer, size, bufferOffset, objectOffset);
}

}}

#endif
