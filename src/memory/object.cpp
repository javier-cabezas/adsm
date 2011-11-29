#include "config/config.h"

#include "core/address_space.h"

#include "Memory.h"
#include "object.h"

namespace __impl { namespace memory {

#ifdef DEBUG
Atomic object::Id_ = 0;
#endif

object::~object()
{
    vector_block::iterator i;
    lock_write();
    gmacError_t ret = coherenceOp(&protocol_interface::deleteBlock);
    ASSERTION(ret == gmacSuccess);
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
object::coherenceOp(gmacError_t (protocol_interface::*f)(block_ptr))
{
    gmacError_t ret = gmacSuccess;
    vector_block::const_iterator i;
    for(i = blocks_.begin(); i != blocks_.end(); ++i) {
        //ret = (*i)->coherenceOp(f);
        ret = (protocol_.*f)(*i);
        if(ret != gmacSuccess) break;
    }
    return ret;
}

gmacError_t
object::to_io_device(hal::device_output &output, size_t objOff, size_t count)
{
    hal::event_t event;
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
    hal::event_t event;
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
gmacError_t object::memoryOp(protocol_interface::MemoryOp op,
                             core::io_buffer &buffer, size_t size, size_t bufferOffset, size_t objectOffset)
{
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    vector_block::const_iterator i = get_block(objectOffset, blockOffset);
    for(; i != blocks_.end() && size > 0; i++) {
        block &block = *i->second;
        size_t blockSize = block.size() - blockOffset;
        blockSize = size < blockSize? size: blockSize;
        buffer.wait();
        ret = block.memoryOp(op, buffer, blockSize, bufferOffset, blockOffset);
        blockOffset = 0;
        bufferOffset += blockSize;
        size -= blockSize;
    }
    return ret;
}
#endif

gmacError_t object::memset(size_t offset, int v, size_t size)
{
	hal::event_t event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    const_locked_iterator i = get_block(offset, &blockOffset);
    for(; i != blocks_.end() && size > 0; ++i) {
        block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = size < blockSize? size: blockSize;
        block->lock();
        event = protocol_.memset(block, blockOffset, v, blockSize, ret);
        block->unlock();
        blockOffset = 0;
        size -= blockSize;
    }
    return ret;
}

gmacError_t
object::memcpyToObject(size_t objOff, const hostptr_t src, size_t size)
{
    hal::event_t event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    size_t off = 0;
    const_locked_iterator i = get_block(objOff, &blockOffset);
    for(; i != blocks_.end() && size > 0; ++i) {
        block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = size < blockSize? size: blockSize;
        block->lock();
        event = protocol_.copyToBlock(block, blockOffset,
                                             src + off,
                                             blockSize, ret);
        block->unlock();
        //block.memoryOp(op, buffer, blockSize, bufferOffset, blockOffset);
        blockOffset = 0;
        off  += blockSize;
        size -= blockSize;
    }
    return ret;
}

gmacError_t
object::memcpyObjectToObject(object &dstObj, size_t dstOffset, size_t srcOffset, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = gmacSuccess;

    hal::event_t event;

    const_locked_iterator i = get_block(srcOffset);
    TRACE(LOCAL, "FP: %p "FMT_SIZE, dstObj.addr() + dstOffset, size);
    const_locked_iterator j = dstObj.get_block(dstOffset);
    TRACE(LOCAL, "FP: %p vs %p "FMT_SIZE, (*j)->addr(), dstObj.addr() + dstOffset, size);
    size_t left = size;
    while (left > 0) {
        size_t copySize = left < dstObj.blockEnd(dstOffset)? left: dstObj.blockEnd(dstOffset);
        // Single copy from the source to fill the buffer
        if (copySize <= blockEnd(srcOffset)) {
            TRACE(LOCAL, "FP: Copying1: "FMT_SIZE" bytes", copySize);
            event = protocol_.copyBlockToBlock(*j, dstOffset % blockSize(),
                                               *i, srcOffset % blockSize(), copySize, ret);
            ASSERTION(ret == gmacSuccess);
            ++i;
        }
        else { // Two copies from the source to fill the buffer
            TRACE(LOCAL, "FP: Copying2: "FMT_SIZE" bytes", copySize);
            size_t firstCopySize = blockEnd(srcOffset);
            size_t secondCopySize = copySize - firstCopySize;

            event = protocol_.copyBlockToBlock(*j, dstOffset % blockSize(),
                                               *i, srcOffset % blockSize(),
                                               firstCopySize, ret);
            ASSERTION(ret == gmacSuccess);
            ++i;
            event = protocol_.copyBlockToBlock(*j, (dstOffset + firstCopySize) % blockSize(),
                                               *i, (srcOffset + firstCopySize) % blockSize(),
                                               secondCopySize, ret);
            ASSERTION(ret == gmacSuccess);
        }
        left -= copySize;
        dstOffset += copySize;
        srcOffset += copySize;
        ++j;
    }

    if (event.is_valid()) {
        ret = event.sync();
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

    size_t bufSize = size < dstObj.blockSize()? size: dstObj.blockSize();
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
            copySize = (left < dstObj.blockSize()) ? left: dstObj.blockSize();
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
object::memcpyFromObject(hostptr_t dst, size_t objOff, size_t size)
{
    hal::event_t event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    size_t off = 0;
    const_locked_iterator i = get_block(objOff);
    for(; i != blocks_.end() && size > 0; ++i) {
        block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = size < blockSize? size: blockSize;
        event = protocol_.copyFromBlock(dst + off,
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

    size_t bufSize = size < blockSize()? size: blockSize();
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

gmacError_t
object::signal_read(hostptr_t addr)
{
    gmacError_t ret = gmacSuccess;
    lock_read();
    /// \todo is this validate necessary?
    //validate();
    const_locked_iterator i = get_block(addr - addr_);
    if(i == blocks_.end()) ret = gmacErrorInvalidValue;
    else if((*i)->addr() > addr) ret = gmacErrorInvalidValue;
    else ret = protocol_.signal_read(*i, addr);
    unlock();
    return ret;
}

gmacError_t
object::signal_write(hostptr_t addr)
{
    gmacError_t ret = gmacSuccess;
    lock_read();
    modifiedObject();
    const_locked_iterator i = get_block(addr - addr_);
    if(i == blocks_.end()) ret = gmacErrorInvalidValue;
    else if((*i)->addr() > addr) ret = gmacErrorInvalidValue;
    else ret = protocol_.signal_write(*i, addr);
    unlock();
    return ret;
}

gmacError_t
object::dump(std::ostream &out, protocol::common::Statistic stat)
{
#ifdef DEBUG
    lock_write();
    std::ostringstream oss;
    oss << (void *) addr();
    out << oss.str() << " ";
    gmacError_t ret = forEachBlock(&protocol_interface::dump, out, stat);
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
