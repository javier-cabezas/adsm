#include "config/config.h"

#include "core/address_space.h"

#include "memory.h"
#include "object.h"

namespace __impl { namespace memory {

#ifdef DEBUG
Atomic object::Id_ = 0;
#endif

object::~object()
{
    vector_block::iterator i;
    lock_write();
    gmacError_t err;
    hal::event_ptr evt;
    evt = coherence_op(&protocol::remove_block, err);
    ASSERTION(err == gmacSuccess);
    blocks_.clear();
    unlock();
}

object::const_locked_iterator
object::get_block(size_t objectOffset, size_t *blockOffset) const
{
	if (objectOffset > size()) {
		return const_locked_iterator(blocks_.end(), blocks_);
	}

	if (blockOffset != NULL) {
		*blockOffset = objectOffset % config::params::BlockSize;
	}

	return const_locked_iterator(blocks_.begin() + (objectOffset / config::params::BlockSize),
			                     blocks_);
}

gmacError_t
object::to_io_device(hal::device_output &output, size_t objOff, size_t count)
{
    hal::event_ptr event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    size_t off = 0;
    const_locked_iterator i = get_block(objOff, &blockOffset);
    for(; count > 0 && i != blocks_.end(); ++i) {
        block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = count < blockSize? count: blockSize;
        event = protocol_.to_io_device(output,
                                       block, blockOffset,
                                       blockSize, ret);
        //block.memoryOp(op, buffer, blockSize, bufferOffset, blockOffset);
        blockOffset = 0;
        off   += blockSize;
        count -= blockSize;
    }
    return ret;
}

gmacError_t
object::from_io_device(size_t objOff, hal::device_input &input, size_t count)
{
    hal::event_ptr event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    size_t off = 0;
    const_locked_iterator i = get_block(objOff, &blockOffset);
    for(; i != blocks_.end() && count > 0; ++i) {
        block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = count < blockSize? count: blockSize;
        event = protocol_.from_io_device(block, blockOffset,
                                         input,
                                         blockSize, ret);
        //block.memoryOp(op, buffer, blockSize, bufferOffset, blockOffset);
        blockOffset = 0;
        off   += blockSize;
        count -= blockSize;
    }
    return ret;

}

#if 0
gmacError_t object::memoryOp(protocol::MemoryOp op,
                             core::io_buffer &buffer, size_t size, size_t bufferOffset, size_t objectOffset)
{
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    vector_block::const_iterator i = get_block(objectOffset, blockOffset);
    for(; i != blocks_.end() && size > 0; i++) {
        block &block = *i->second;
        size_t get_block_size = block.size() - blockOffset;
        get_block_size = size < get_block_size? size: get_block_size;
        buffer.wait();
        ret = block.memoryOp(op, buffer, get_block_size, bufferOffset, blockOffset);
        blockOffset = 0;
        bufferOffset += get_block_size;
        size -= get_block_size;
    }
    return ret;
}
#endif

gmacError_t object::memset(size_t offset, int v, size_t size)
{
	hal::event_ptr event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    const_locked_iterator i = get_block(offset, &blockOffset);
    for(; i != blocks_.end() && size > 0; ++i) {
        block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = size < blockSize? size: blockSize;
        event = protocol_.memset(block, blockOffset, v, blockSize, ret);
        blockOffset = 0;
        size -= blockSize;
    }
    return ret;
}

gmacError_t
object::memcpy_to_object(size_t objOff, const hostptr_t src, size_t size)
{
    hal::event_ptr event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    size_t off = 0;
    const_locked_iterator i = get_block(objOff, &blockOffset);
    for(; i != blocks_.end() && size > 0; ++i) {
        block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = size < blockSize? size: blockSize;
        event = protocol_.copy_to_block(block, blockOffset,
                                             src + off,
                                             blockSize, ret);
        //block.memoryOp(op, buffer, blockSize, bufferOffset, blockOffset);
        blockOffset = 0;
        off  += blockSize;
        size -= blockSize;
    }
    return ret;
}

gmacError_t
object::memcpy_object_to_object(object &dstObj, size_t dstOffset, size_t srcOffset, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    hal::event_ptr event;

    const_locked_iterator i = get_block(srcOffset);
    TRACE(LOCAL, "FP: %p "FMT_SIZE, dstObj.addr() + dstOffset, size);
    const_locked_iterator j = dstObj.get_block(dstOffset);
    TRACE(LOCAL, "FP: %p vs %p "FMT_SIZE, (*j)->addr(), dstObj.addr() + dstOffset, size);
    size_t left = size;
    while (left > 0) {
        size_t copySize = left < dstObj.blockEnd(dstOffset)? left: dstObj.blockEnd(dstOffset);
        // Single copy from the source to copy the block
        if (copySize <= blockEnd(srcOffset)) {
            TRACE(LOCAL, "FP: Copying1: "FMT_SIZE" bytes", copySize);
            event = protocol_.copy_block_to_block(*j, dstOffset % get_block_size(),
                                               *i, srcOffset % get_block_size(), copySize, ret);
            ASSERTION(ret == gmacSuccess);
            ++i;
        }
        else { // Two copies from the source to copy the block
            TRACE(LOCAL, "FP: Copying2: "FMT_SIZE" bytes", copySize);
            size_t firstCopySize = blockEnd(srcOffset);
            size_t secondCopySize = copySize - firstCopySize;

            event = protocol_.copy_block_to_block(*j, dstOffset % get_block_size(),
                                               *i, srcOffset % get_block_size(),
                                               firstCopySize, ret);
            ASSERTION(ret == gmacSuccess);
            ++i;
            event = protocol_.copy_block_to_block(*j, (dstOffset + firstCopySize) % get_block_size(),
                                               *i, (srcOffset + firstCopySize) % get_block_size(),
                                               secondCopySize, ret);
            ASSERTION(ret == gmacSuccess);
        }
        left -= copySize;
        dstOffset += copySize;
        srcOffset += copySize;
        ++j;
    }

    if (event) {
        ret = event->sync();
    }

#if 0
        dstObj.unlock();
        unlock();
#endif

    trace::ExitCurrentFunction();
    return ret;

#if 0
    // We need to I/O buffers to double-buffer the copy
    core::io_buffer *active;
    core::io_buffer *passive;

    // Control variables
    size_t left = size;

    // Adjust the first copy to deal with a single block
    size_t copySize = size < dstObj.blockEnd(dstOffset)? size: dstObj.blockEnd(dstOffset);

    size_t bufSize = size < dstObj.get_block_size()? size: dstObj.get_block_size();
    active = owner().create_io_buffer(bufSize, GMAC_PROT_READWRITE);
    ASSERTION(bufSize >= copySize);

    if (copySize < size) {
        passive = owner().create_io_buffer(bufSize, GMAC_PROT_READWRITE);
    } else {
        passive = NULL;
    }

    // Single copy from the source to fill the buffer
    if (copySize <= blockEnd(srcOffset)) {
        ret = copyToBuffer(*active, copySize, 0, srcOffset);
        ASSERTION(ret == gmacSuccess);
    }
    else { // Two copies from the source to fill the buffer
        size_t firstCopySize = blockEnd(srcOffset);
        size_t secondCopySize = copySize - firstCopySize;
        ASSERTION(bufSize >= firstCopySize + secondCopySize);

        ret = copyToBuffer(*active, firstCopySize, 0, srcOffset);
        ASSERTION(ret == gmacSuccess);
        ret = copyToBuffer(*active, secondCopySize, firstCopySize, srcOffset + firstCopySize);
        ASSERTION(ret == gmacSuccess);
    }

    // Copy first chunk of data
    while(left > 0) {
        active->wait(); // Wait for the active buffer to be full
        ret = dstObj.copyFromBuffer(*active, copySize, 0, dstOffset);
        if(ret != gmacSuccess) {
            trace::ExitCurrentFunction();
            return ret;
        }
        left -= copySize;
        srcOffset += copySize;
        dstOffset += copySize;
        if(left > 0) {
            copySize = (left < dstObj.get_block_size()) ? left: dstObj.get_block_size();
            ASSERTION(bufSize >= copySize);
            // Avoid overwritting a buffer that is already in use
            passive->wait();

            // Request the next copy
            // Single copy from the source to fill the buffer
            if (copySize <= blockEnd(srcOffset)) {
                ret = copyToBuffer(*passive, copySize, 0, srcOffset);
                ASSERTION(ret == gmacSuccess);
            }
            else { // Two copies from the source to fill the buffer
                size_t firstCopySize = blockEnd(srcOffset);
                size_t secondCopySize = copySize - firstCopySize;
                ASSERTION(bufSize >= firstCopySize + secondCopySize);

                ret = copyToBuffer(*passive, firstCopySize, 0, srcOffset);
                ASSERTION(ret == gmacSuccess);
                ret = copyToBuffer(*passive, secondCopySize, firstCopySize, srcOffset + firstCopySize);
                ASSERTION(ret == gmacSuccess);
            }

            // Swap buffers
            core::io_buffer *tmp = active;
            active = passive;
            passive = tmp;
        }
    }
    // Clean up buffers after they are idle
    if (passive != NULL) {
        passive->wait();
        owner().destroy_io_buffer(*passive);
    }
    if (active  != NULL) {
        active->wait();
        owner().destroy_io_buffer(*active);
    }

    trace::ExitCurrentFunction();
    return ret;
#endif
}

gmacError_t
object::memcpy_from_object(hostptr_t dst, size_t objOff, size_t size)
{
    hal::event_ptr event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    size_t off = 0;
    const_locked_iterator i = get_block(objOff);
    for(; i != blocks_.end() && size > 0; ++i) {
        block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = size < blockSize? size: blockSize;
        event = protocol_.copy_from_block(dst + off,
                                        block, blockOffset,
                                        blockSize, ret);
        //block.memoryOp(op, buffer, blockSize, bufferOffset, blockOffset);
        blockOffset = 0;
        off  += blockSize;
        size -= blockSize;
    }
    return ret;

#if 0
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    // We need to I/O buffers to double-buffer the copy
    core::io_buffer *active;
    core::io_buffer *passive;

    // Control variables
    size_t left = size;

    // Adjust the first copy to deal with a single block
    size_t copySize = size < blockEnd(objOff)? size: blockEnd(objOff);

    size_t bufSize = size < get_block_size()? size: get_block_size();
    active = owner().create_io_buffer(bufSize, GMAC_PROT_READ);
    ASSERTION(bufSize >= copySize);

    if (copySize < size) {
        passive = owner().create_io_buffer(bufSize, GMAC_PROT_READ);
    } else {
        passive = NULL;
    }

    // Copy the data to the first block
    ret = copyToBuffer(*active, copySize, 0, objOff);
    ASSERTION(ret == gmacSuccess);
    //if(ret != gmacSuccess) return ret;
    while(left > 0) {
        // Save values to use when copying the buffer to host memory
        size_t previousCopySize = copySize;
        left      -= copySize;
        objOff += copySize;
        if(left > 0) {
            // Start copying data from host memory to the passive I/O buffer
            copySize = (left < passive->size()) ? left : passive->size();
            ASSERTION(bufSize >= copySize);
            // No need to wait for the buffer, because ::memcpy is a
            // synchronous call
            ret = copyToBuffer(*passive, copySize, 0, objOff);
            ASSERTION(ret == gmacSuccess);
        }
        // Wait for the active buffer to be full
        active->wait();
        // Copy the active buffer to host
        ::memcpy(dst, active->addr(), previousCopySize);
        dst += previousCopySize;

        // Swap buffers
        core::io_buffer *tmp = active;
        active = passive;
        passive = tmp;
    }
    // No need to wait for the buffers because we waited for them before ::memcpy
    if (passive != NULL) owner().destroy_io_buffer(*passive);
    if (active  != NULL) owner().destroy_io_buffer(*active);

    trace::ExitCurrentFunction();
    return ret;
#endif
}

hal::event_ptr
object::signal_read(hostptr_t addr, gmacError_t &err)
{
    hal::event_ptr ret;
    lock_read();
    /// \todo is this validate necessary?
    //validate();
    const_locked_iterator i = get_block(addr - addr_);
    if(i == blocks_.end()) err = gmacErrorInvalidValue;
    else if((*i)->addr() > addr) err = gmacErrorInvalidValue;
    else ret = protocol_.signal_read(*i, addr, err);
    unlock();
    return ret;
}

hal::event_ptr
object::signal_write(hostptr_t addr, gmacError_t &err)
{
    hal::event_ptr ret;
    lock_read();
    modified_object();
    const_locked_iterator i = get_block(addr - addr_);
    if(i == blocks_.end()) err = gmacErrorInvalidValue;
    else if((*i)->addr() > addr) err = gmacErrorInvalidValue;
    else ret = protocol_.signal_write(*i, addr, err);
    unlock();
    return ret;
}

gmacError_t
object::dump(std::ostream &out, protocols::common::Statistic stat)
{
#ifdef DEBUG
    lock_write();
    std::ostringstream oss;
    oss << (void *) addr();
    out << oss.str() << " ";
    gmacError_t ret = gmacSuccess;
    std::for_each(blocks_.begin(),
    		      blocks_.end(), [&protocol_, &out, stat, &ret](block_ptr ptr)
    		                     {
    	                             if (ret == gmacSuccess) {
    	                            	 ret = protocol_.dump(ptr, out, stat);
    	                             }
    		                     });
    out << std::endl;
    unlock();
    if (dumps_.find(stat) == dumps_.end()) dumps_[stat] = 0;
    dumps_[stat]++;
#else
    gmacError_t ret = gmacSuccess;
#endif
    return ret;
}

}}
