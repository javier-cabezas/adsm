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
    BlockMap::iterator i;
    lockWrite();
    for(i = blocks_.begin(); i != blocks_.end(); i++)
        i->second->release();
    blocks_.clear();
    unlock();
    Memory::unshadow(shadow_, size_);
}

inline uint8_t *Object::addr() const
{
    // No need for lock -- addr_ is never modified
    return (uint8_t *) addr_;   
}

inline uint8_t *Object::end() const
{
    // No need for lock -- addr_ and size_ are never modified
    return addr_ + size_;
}

inline size_t Object::size() const
{
    // No need for lock -- size_ is never modified
    return size_;
}

inline bool Object::valid() const
{
    // No need for lock -- valid_ is never modified
	return valid_;
}

inline gmacError_t Object::acquire() const
{
    lockRead();
	gmacError_t ret = coherenceOp(&Protocol::acquire);
    unlock();
    return ret;
}

inline gmacError_t Object::toHost() const
{
	lockRead();
    gmacError_t ret= coherenceOp(&Protocol::toHost);
    unlock();
    return ret;
}

inline gmacError_t Object::toDevice() const
{
    lockRead();
	gmacError_t ret = coherenceOp(&Protocol::toDevice);
    unlock();
    return ret;
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
    lockRead();
	gmacError_t ret = memoryOp(&Protocol::copyToBuffer, buffer, size, bufferOffset, objectOffset);
    unlock();
    return ret;
}

inline gmacError_t Object::copyFromBuffer(core::IOBuffer &buffer, size_t size, 
										unsigned bufferOffset, unsigned objectOffset) const
{
    lockRead();
	gmacError_t ret = memoryOp(&Protocol::copyFromBuffer, buffer, size, bufferOffset, objectOffset);
    unlock();
    return ret;
}

}}

#endif
