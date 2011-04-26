#include "config/config.h"
#include "core/IOBuffer.h"

#include "Object.h"
#include "memory/Memory.h"

namespace __impl { namespace memory {

#ifdef DEBUG
Atomic Object::Id_ = 0;
#endif

Object::BlockMap::const_iterator Object::firstBlock(size_t &objectOffset) const
{
    BlockMap::const_iterator i = blocks_.begin();
    if(i == blocks_.end()) return i;
    while(objectOffset >= i->second->size()) {
        objectOffset -= i->second->size();
        i++;
        if(i == blocks_.end()) return i;
    }
    return i;
}

gmacError_t Object::coherenceOp(gmacError_t (Protocol::*f)(Block &))
{
	gmacError_t ret = gmacSuccess;
	BlockMap::const_iterator i;
	for(i = blocks_.begin(); i != blocks_.end(); i++) {
		ret = i->second->coherenceOp(f);
		if(ret != gmacSuccess) break;
	}
	return ret;
}

#if 0
gmacError_t Object::memoryOp(Protocol::MemoryOp op, core::IOBuffer &buffer, size_t size, 
							 size_t bufferOffset, size_t objectOffset) const
#endif
gmacError_t Object::memoryOp(Protocol::MemoryOp op,
                             core::IOBuffer &buffer, size_t size, size_t bufferOffset, size_t objectOffset)
{
	gmacError_t ret = gmacSuccess;
	BlockMap::const_iterator i = firstBlock(objectOffset); // objectOffset gets modified
    size_t blockOffset = objectOffset % blockSize();
	for(; i != blocks_.end() && size > 0; i++) {
		size_t blockSize = i->second->size() - blockOffset;
		blockSize = size < blockSize? size: blockSize;
        buffer.wait();
		ret = i->second->memoryOp(op, buffer, blockSize, bufferOffset, blockOffset);
		blockOffset = 0;
		bufferOffset += blockSize;
		size -= blockSize;
	}
	return ret;
}


gmacError_t Object::memset(size_t offset, int v, size_t size)
{
    gmacError_t ret = gmacSuccess;
    BlockMap::const_iterator i = firstBlock(offset); // offset gets modified
    size_t blockOffset = offset % blockSize();
	for(; i != blocks_.end() && size > 0; i++) {
		size_t blockSize = i->second->size() - blockOffset;
		blockSize = size < blockSize? size: blockSize;
		ret = i->second->memset(v, blockSize, blockOffset);
		blockOffset = 0;
		size -= blockSize;
	}
	return ret;
}

}}
