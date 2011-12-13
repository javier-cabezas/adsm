#ifndef GMAC_MEMORY_OBJECT_IMPL_H_
#define GMAC_MEMORY_OBJECT_IMPL_H_

#include <fstream>
#include <sstream>

#include "Protocol.h"
#include "block.h"

#include "util/Logger.h"

namespace __impl { namespace memory {

inline object::object(protocol_interface &protocol, hostptr_t addr, size_t size) :
    Lock("object"),
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

inline void
object::set_last_event(hal::event_ptr event)
{
    if (!event) return;

    switch (event->get_type()) {
    case hal::event_ptr::type::TransferToHost:
        lastToHost_ = event;
        break;
    case hal::event_ptr::type::TransferToDevice:
        lastToDevice_ = event;
        break;
    case hal::event_ptr::type::TransferHost:
        lastHost_ = event;
        break;
    case hal::event_ptr::type::TransferDevice:
        lastDevice_ = event;
        break;
    case hal::event_ptr::type::Kernel:
        lastKernel_ = event;
        break;
    case hal::event_ptr::type::Invalid:
        break;
    }

    last_ = event;
}

inline hal::event_ptr
object::get_last_event(hal::event_ptr::type type) const
{
    switch (type) {
    case hal::event_ptr::type::TransferToHost:
        return lastToHost_;
    case hal::event_ptr::type::TransferToDevice:
        return lastToDevice_;
    case hal::event_ptr::type::TransferHost:
        return lastHost_;
    case hal::event_ptr::type::TransferDevice:
        return lastDevice_;
    case hal::event_ptr::type::Kernel:
        return lastKernel_;
    case hal::event_ptr::type::Invalid:
        return hal::event_ptr();
    }
}

inline
protocol_interface &
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
hal::event_ptr
object::coherenceOp(hal::event_ptr (protocol_interface::*op)(block_ptr, T &, gmacError_t &),
                    T &param,
                    gmacError_t &err)
{
    hal::event_ptr ret;
    vector_block::const_iterator i;
    for(i = blocks_.begin(); i != blocks_.end(); i++) {
        ret = (protocol_.*op)(*i, param, err);
        if(err != gmacSuccess) break;
        set_last_event(ret);
    }
    return ret;
}

inline hal::event_ptr
object::acquire(GmacProtection &prot, gmacError_t &err)
{
    lock_write();
    hal::event_ptr ret;
    TRACE(LOCAL, "Acquiring object %p?", addr_);
    if (released_ == true) {
        TRACE(LOCAL, "Acquiring object %p", addr_);
        ret = coherenceOp<GmacProtection>(&protocol_interface::acquire, prot, err);
        if (err == gmacSuccess) {
            set_last_event(ret);
        }
    }
    released_ = false;
    unlock();
    return ret;
}

inline hal::event_ptr 
object::release(gmacError_t &err)
{
    hal::event_ptr ret;
    lock_write();
    released_ = true;
    unlock();
    err = gmacSuccess;
    return ret;
}

inline bool
object::is_released() const
{
    bool ret;
    lock_read();
    ret = released_;
    unlock();
    return ret;
}

inline hal::event_ptr
object::releaseBlocks(gmacError_t &err)
{
    lock_write();
    hal::event_ptr ret;

    TRACE(LOCAL, "Releasing object %p?", addr_);
    if (released_ == false) {
        TRACE(LOCAL, "Releasing object %p", addr_);
        ret = coherenceOp(&protocol_interface::release, err);
    }

    released_ = true;
    unlock();
    return ret;
}

#ifdef USE_VM
inline gmacError_t
object::acquireWithBitmap()
{
    lock_read();
    gmacError_t ret = coherenceOp(&protocol_interface::acquireWithBitmap);
    unlock();
    return ret;
}
#endif

template <typename P1, typename P2>
gmacError_t
object::forEachBlock(gmacError_t (protocol_interface::*op)(block_ptr, P1 &, P2), P1 &p1, P2 p2)
{
    lock_read();
    gmacError_t ret = gmacSuccess;
    vector_block::iterator i;
    for(i = blocks_.begin(); i != blocks_.end(); i++) {
        //ret = ((*i)->*f)(p1, p2);
        ret = (protocol_.*op)(*i, p1, p2);
        // TODO remove this if the code is no longer executed
        abort();
    }
    unlock();
    return ret;
}

inline hal::event_ptr 
object::toHost(gmacError_t &err)
{
    lock_read();
    hal::event_ptr ret = coherenceOp(&protocol_interface::toHost, err);
    if (err == gmacSuccess) {
        set_last_event(ret);
    }
    unlock();
    return ret;
}

inline hal::event_ptr
object::toAccelerator(gmacError_t &err)
{
    lock_read();
    hal::event_ptr ret = coherenceOp(&protocol_interface::release, err);
    if (err == gmacSuccess) {
        set_last_event(ret);
    }
    unlock();
    return ret;
}

#if 0
inline gmacError_t
object::copyToBuffer(core::io_buffer &buffer, size_t size,
                     size_t bufferOffset, size_t objectOffset)
{
    lock_read();
    gmacError_t ret = memoryOp(&protocol_interface::copyToBuffer, buffer, size,
                               bufferOffset, objectOffset);
    unlock();
    return ret;
}

inline gmacError_t object::copyFromBuffer(core::io_buffer &buffer, size_t size,
                                          size_t bufferOffset, size_t objectOffset)
{
    lock_read();
    gmacError_t ret = memoryOp(&protocol_interface::copyFromBuffer, buffer, size,
                               bufferOffset, objectOffset);
    unlock();
    return ret;
}
#endif

#if 0
inline gmacError_t object::copyObjectToObject(object &dst, size_t dstOff,
                                              object &src, size_t srcOff, size_t count)
{
    dst.lock_write();
    src.lock_write();
        gmacError_t ret = memoryOp(&protocol_interface::copyFromBuffer, buffer, size,
        bufferOffset, objectOffset);
    dst.unlock();
    src.unlock();
    return ret;
}
#endif

}}

#endif
