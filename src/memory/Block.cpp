#include "Block.h"
#include "Object.h"


namespace __impl { namespace memory {

gmacError_t Block::memcpyFromObject(const Object &object, size_t size, 
                                    unsigned blockOffset, unsigned objectOffset)
{
    gmacError_t ret = gmacSuccess;
    lock();
    // Update the contents of host memory, if necessary
    if(blockOffset > 0 || size < size_) ret = protocol_.toHost(*this);
    if(ret != gmacSuccess) goto exit_func;
    
    // Copy the data from the source object
    ret = object.memcpyToMemory(shadow_ + blockOffset, object.addr() + objectOffset, size);
    if(ret != gmacSuccess) goto exit_func;

    // Trigger a fake signal to set the block as dirty
    ret = protocol_.signalWrite(*this);

exit_func:
    unlock();
    return ret;
}

}}

