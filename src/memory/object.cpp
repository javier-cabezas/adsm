#include "config/config.h"

#include "address_space.h"
#include "memory.h"
#include "object.h"

namespace __impl { namespace memory {

object::~object()
{
    vector_block::iterator i;
    gmacError_t err;
    hal::event_ptr evt;
    evt = coherence_op(&protocol::remove_block, err);
    ASSERTION(err == gmacSuccess);
    blocks_.clear();
}

object::const_locking_iterator
object::get_block(size_t objectOffset, size_t *blockOffset) const
{
	if (objectOffset > size()) {
		return const_locking_iterator(blocks_.end(), blocks_);
	}

	if (blockOffset != NULL) {
		*blockOffset = objectOffset % config::params::BlockSize;
	}

	return const_locking_iterator(blocks_.begin() + (objectOffset / config::params::BlockSize),
	                              blocks_);
}

object::const_locking_iterator
object::begin() const
{
    return const_locking_iterator(blocks_.begin(), blocks_);
}

object::const_locking_iterator
object::end() const
{
    return const_locking_iterator(blocks_.end(), blocks_);
}

gmacError_t
object::to_io_device(hal::device_output &output, size_t objOff, size_t count)
{
    hal::event_ptr event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    size_t off = 0;
    const_locking_iterator i = get_block(objOff, &blockOffset);
    for(; count > 0 && i != end(); ++i) {
        protocols::common::block_ptr block = *i;
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
    const_locking_iterator i = get_block(objOff, &blockOffset);
    for(; i != end() && count > 0; ++i) {
        protocols::common::block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = count < blockSize? count: blockSize;
        event = protocol_.from_io_device(block, blockOffset,
                                         input,
                                         blockSize, ret);
        blockOffset = 0;
        off   += blockSize;
        count -= blockSize;
    }
    return ret;

}

gmacError_t
object::memset(size_t offset, int v, size_t size)
{
	hal::event_ptr event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    const_locking_iterator i = get_block(offset, &blockOffset);
    for(; i != end() && size > 0; ++i) {
        protocols::common::block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = size < blockSize? size: blockSize;
        event = protocol_.memset(block, blockOffset, v, blockSize, ret);
        blockOffset = 0;
        size -= blockSize;
    }
    return ret;
}

gmacError_t
object::memcpy_to_object(size_t objOff, host_const_ptr src, size_t size)
{
    hal::event_ptr event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    size_t off = 0;
    const_locking_iterator i = get_block(objOff, &blockOffset);
    for(; i != end() && size > 0; ++i) {
        protocols::common::block_ptr block = *i;
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

    const_locking_iterator i = get_block(srcOffset);
    TRACE(LOCAL, "FP: %p "FMT_SIZE, dstObj.get_bounds().start + dstOffset, size);
    const_locking_iterator j = dstObj.get_block(dstOffset);
    TRACE(LOCAL, "FP: %p vs %p "FMT_SIZE, (*j)->get_bounds().start,
                                          dstObj.get_bounds().start + dstOffset, size);
    size_t left = size;
    while (left > 0) {
        size_t copySize = left < dstObj.get_block_end(dstOffset)? left: dstObj.get_block_end(dstOffset);
        // Single copy from the source to copy the block
        if (copySize <= get_block_end(srcOffset)) {
            TRACE(LOCAL, "FP: Copying1: "FMT_SIZE" bytes", copySize);
            event = protocol_.copy_block_to_block(*j, dstOffset % get_block_size(),
                                                  *i, srcOffset % get_block_size(), copySize, ret);
            ASSERTION(ret == gmacSuccess);
            ++i;
        }
        else { // Two copies from the source to copy the block
            TRACE(LOCAL, "FP: Copying2: "FMT_SIZE" bytes", copySize);
            size_t firstCopySize = get_block_end(srcOffset);
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

    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t
object::memcpy_from_object(host_ptr dst, size_t objOff, size_t size)
{
    hal::event_ptr event;
    gmacError_t ret = gmacSuccess;
    size_t blockOffset = 0;
    size_t off = 0;
    for (const_locking_iterator i  = get_block(objOff);
                                i != end() && size > 0;
                              ++i) {
        protocols::common::block_ptr block = *i;
        size_t blockSize = block->size() - blockOffset;
        blockSize = size < blockSize? size: blockSize;
        event = protocol_.copy_from_block(dst + off,
                                          block, blockOffset,
                                          blockSize, ret);
        blockOffset = 0;
        off  += blockSize;
        size -= blockSize;
    }
    return ret;
}

hal::event_ptr
object::signal_read(host_ptr addr, gmacError_t &err)
{
    hal::event_ptr ret;
    /// \todo is this validate necessary?
    //validate();
    const_locking_iterator i = get_block(addr - addr_);
    if (i == end()) err = gmacErrorInvalidValue;
    else if ((*i)->get_bounds().start > addr) err = gmacErrorInvalidValue;
    else ret = protocol_.signal_read(*i, addr, err);
    return ret;
}

hal::event_ptr
object::signal_write(host_ptr addr, gmacError_t &err)
{
    hal::event_ptr ret;
    modified_object();
    const_locking_iterator i = get_block(addr - addr_);
    if(i == end()) err = gmacErrorInvalidValue;
    else if((*i)->get_bounds().start > addr) err = gmacErrorInvalidValue;
    else ret = protocol_.signal_write(*i, addr, err);
    return ret;
}

gmacError_t
object::dump(std::ostream &out, protocols::common::Statistic stat)
{
#ifdef DEBUG
    lock::lock();
    std::ostringstream oss;
    oss << (void *) get_bounds().start;
    out << oss.str() << " ";
    gmacError_t ret;
    for (const_locking_iterator it = begin(); it != end(); ++it) {
        ret = protocol_.dump(*it, out, stat);
    }

    ASSERTION(ret == gmacSuccess);

    out << std::endl;
    if (dumps_.find(stat) == dumps_.end()) dumps_[stat] = 0;
    dumps_[stat]++;
    lock::unlock();
#else
    gmacError_t ret = gmacSuccess;
#endif
    return ret;
}

}}
