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
	// Skip blocks at the begining of the object until we reach 
    // the block where the operation has to be applied for the
    // first time
	BlockMap::const_iterator i = blocks_.begin();
    while(objectOffset >= i->second->size()) {
        objectOffset -= unsigned(i->second->size());
        i++;
        ASSERTION(i != blocks_.end());
    }
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

}}
