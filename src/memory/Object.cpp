#include "config/config.h"

#include "Object.h"
#include "memory/Memory.h"

namespace __impl { namespace memory {

Object::BlockMap::const_iterator Object::firstBlock(unsigned &objectOffset) const
{
    BlockMap::const_iterator i = blocks_.begin();
    if(i == blocks_.end()) return i;
    while(objectOffset >= i->second->size()) {
        objectOffset -= unsigned(i->second->size());
        i++;
        if(i == blocks_.end()) return i;
    }
    return i;
}

gmacError_t Object::coherenceOp(Protocol::CoherenceOp op) const
{
	gmacError_t ret = gmacSuccess;
	BlockMap::const_iterator i;
	for(i = blocks_.begin(); i != blocks_.end(); i++) {
		ret = i->second->coherenceOp(op);
		if(ret != gmacSuccess) break;
	}
	return ret;
}

gmacError_t Object::memoryOp(Protocol::MemoryOp op, core::IOBuffer &buffer, size_t size, 
							 unsigned bufferOffset, unsigned objectOffset) const
{
	gmacError_t ret = gmacSuccess;
	BlockMap::const_iterator i = firstBlock(objectOffset); // objectOffset gets modified
    unsigned blockOffset = objectOffset;
	for(; i != blocks_.end(); i++) {
		size_t blockSize = size - blockOffset;
		blockSize = (blockSize < i->second->size()) ? blockSize : i->second->size();
		ret = i->second->memoryOp(op, buffer, blockSize, bufferOffset, blockOffset);
		blockOffset = 0;
		bufferOffset += unsigned(i->second->size());
		size -= blockSize;
        if(size == 0) break;
	}
	return ret;
}

gmacError_t Object::memset(void *addr, int v, size_t size) const
{
    gmacError_t ret = gmacSuccess;
    unsigned objectOffset = unsigned((uint8_t *)addr - addr_);
    BlockMap::const_iterator i = firstBlock(objectOffset); // objectOffset gets modified
    unsigned blockOffset = objectOffset;
	for(; i != blocks_.end(); i++) {
		size_t blockSize = size - blockOffset;
		blockSize = (blockSize < i->second->size()) ? blockSize : i->second->size();
		ret = i->second->memset(v, blockSize, blockOffset);
		blockOffset = 0;
		size -= blockSize;
        if(size == 0) break;
	}
	return ret;
}

}}
