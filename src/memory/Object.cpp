#include "config/config.h"

#include "Object.h"
#include "memory/Memory.h"

namespace __impl { namespace memory {

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
	unsigned blockOffset = objectOffset;
	BlockMap::const_iterator i;
	for(i = blocks_.begin(); i != blocks_.end(); i++) {
		size_t blockSize = size - blockOffset;
		blockSize = (blockSize < i->second->size()) ? blockSize : i->second->size();
		ret = i->second->memoryOp(op, buffer, blockSize, bufferOffset, blockOffset);
		blockOffset = 0;
		bufferOffset += unsigned(i->second->size());
		size -= blockSize;
	}
	return ret;
}

}}
