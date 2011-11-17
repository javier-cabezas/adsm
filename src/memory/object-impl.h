#ifndef GMAC_MEMORY_OBJECT_IMPL_H_
#define GMAC_MEMORY_OBJECT_IMPL_H_

#include <fstream>
#include <sstream>

#include "Protocol.h"
#include "Block.h"

#include "util/Logger.h"

namespace __impl { namespace memory {

inline object::object(Protocol &protocol, hostptr_t addr, size_t size) :
    gmac::util::lock_rw("Object"),
    util::Reference("Object"),
    protocol_(protocol),
    addr_(addr),
    size_(size),
    released_(false)
{
#ifdef DEBUG
    id_ = AtomicInc(object::Id_);
#endif
}

#ifdef DEBUG
inline unsigned
object::getId() const
{
    return id_;
}

inline unsigned
object::getDumps(protocol::common::Statistic stat)
{
    if (dumps_.find(stat) == dumps_.end()) dumps_[stat] = 0;
    return dumps_[stat];
}
#endif

inline
Protocol &
object::getProtocol()
{
    return protocol_;
}

inline hostptr_t
object::addr() const
{
    // No need for lock -- addr_ is never modified
    return addr_;
}

inline hostptr_t
object::end() const
{
    // No need for lock -- addr_ and size_ are never modified
    return addr_ + size_;
}

inline ssize_t
object::blockBase(size_t offset) const
{
    return -1 * (offset % blockSize());
}

inline size_t
object::blockEnd(size_t offset) const
{
    if (offset + blockBase(offset) + blockSize() > size_)
        return size_ - offset;
    else
        return size_t(ssize_t(blockSize()) + blockBase(offset));
}

inline size_t
object::blockSize() const
{
    return BlockSize_;
}

inline size_t
object::size() const
{
    // No need for lock -- size_ is never modified
    return size_;
}

template <typename T>
gmacError_t object::coherenceOp(gmacError_t (Protocol::*op)(block_ptr, T &), T &param)
{
    gmacError_t ret = gmacSuccess;
    vector_block::const_iterator i;
    for(i = blocks_.begin(); i != blocks_.end(); i++) {
        ret = (protocol_.*op)(*i, param);
        if(ret != gmacSuccess) break;
    }
    return ret;
}

inline gmacError_t
object::acquire(GmacProtection &prot)
{
    lockWrite();
    gmacError_t ret = gmacSuccess;
    TRACE(LOCAL, "Acquiring object %p?", addr_);
    if (released_ == true) {
        TRACE(LOCAL, "Acquiring object %p", addr_);
        ret = coherenceOp<GmacProtection>(&Protocol::acquire, prot);
    }
    released_ = false;
    unlock();
    return ret;
}

inline gmacError_t
object::release()
{
    lockWrite();
    released_ = true;
    unlock();
    return gmacSuccess;
}

inline gmacError_t
object::releaseBlocks()
{
    lockWrite();
    gmacError_t ret = gmacSuccess;

    TRACE(LOCAL, "Releasing object %p?", addr_);
    if (released_ == false) {
        TRACE(LOCAL, "Releasing object %p", addr_);
        ret = coherenceOp(&Protocol::release);
    }

    released_ = true;
    unlock();
    return ret;
}

#ifdef USE_VM
inline gmacError_t
object::acquireWithBitmap()
{
    lockRead();
    gmacError_t ret = coherenceOp(&Protocol::acquireWithBitmap);
    unlock();
    return ret;
}
#endif

template <typename P1, typename P2>
gmacError_t
object::forEachBlock(gmacError_t (Protocol::*op)(block_ptr, P1 &, P2), P1 &p1, P2 p2)
{
    lockRead();
    gmacError_t ret = gmacSuccess;
    vector_block::iterator i;
    for(i = blocks_.begin(); i != blocks_.end(); i++) {
        //ret = ((*i)->*f)(p1, p2);
        ret = (protocol_.*op)(*i, p1, p2);
    }
    unlock();
    return ret;
}

inline gmacError_t object::toHost()
{
    lockRead();
    gmacError_t ret= coherenceOp(&Protocol::toHost);
    unlock();
    return ret;
}

inline gmacError_t object::toAccelerator()
{
    lockRead();
    gmacError_t ret = coherenceOp(&Protocol::release);
    unlock();
    return ret;
}

#if 0
inline gmacError_t
object::copyToBuffer(core::io_buffer &buffer, size_t size,
                     size_t bufferOffset, size_t objectOffset)
{
    lockRead();
    gmacError_t ret = memoryOp(&Protocol::copyToBuffer, buffer, size,
                               bufferOffset, objectOffset);
    unlock();
    return ret;
}

inline gmacError_t object::copyFromBuffer(core::io_buffer &buffer, size_t size,
                                          size_t bufferOffset, size_t objectOffset)
{
    lockRead();
    gmacError_t ret = memoryOp(&Protocol::copyFromBuffer, buffer, size,
                               bufferOffset, objectOffset);
    unlock();
    return ret;
}
#endif

#if 0
inline gmacError_t object::copyObjectToObject(object &dst, size_t dstOff,
                                              object &src, size_t srcOff, size_t count)
{
    dst.lockWrite();
    src.lockWrite();
        gmacError_t ret = memoryOp(&Protocol::copyFromBuffer, buffer, size,
        bufferOffset, objectOffset);
    dst.unlock();
    src.unlock();
    return ret;
}
#endif

}}

#endif
