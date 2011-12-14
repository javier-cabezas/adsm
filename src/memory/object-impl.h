#ifndef GMAC_MEMORY_OBJECT_IMPL_H_
#define GMAC_MEMORY_OBJECT_IMPL_H_

#include <fstream>
#include <sstream>

#include "protocol.h"
#include "block.h"

#include "trace/logger.h"

namespace __impl { namespace memory {

inline object::object(protocol &protocol, hostptr_t addr, size_t size) :
    Lock("object"),
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
object::getDumps(protocols::common::Statistic stat)
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
protocol &
object::get_protocol()
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
    return -1 * (offset % get_block_size());
}

inline size_t
object::blockEnd(size_t offset) const
{
    if (offset + blockBase(offset) + get_block_size() > size_)
        return size_ - offset;
    else
        return size_t(ssize_t(get_block_size()) + blockBase(offset));
}

inline size_t
object::get_block_size() const
{
    return BlockSize_;
}

inline size_t
object::size() const
{
    // No need for lock -- size_ is never modified
    return size_;
}

template <typename... Args>
hal::event_ptr
object::coherence_op(hal::event_ptr (protocol::*op)(block_ptr, Args..., gmacError_t &),
                     Args... args,
                     gmacError_t &err)
{
    hal::event_ptr ret;
    for(const_locked_iterator i  = get_block(0, NULL);
                              i != blocks_.end();
    		                ++i) {
        ret = (protocol_.*op)(*i, args..., err);
        if (err != gmacSuccess) break;
        set_last_event(ret);
    }
    return ret;
}

inline hal::event_ptr
object::coherence_op(hal::event_ptr (protocol::*f)(block_ptr, gmacError_t &), gmacError_t &err)
{
    hal::event_ptr ret;
    for (const_locked_iterator i  = get_block(0, NULL);
                               i != blocks_.end();
                             ++i) {
        ret = (protocol_.*f)(*i, err);
        if (err != gmacSuccess) break;
        set_last_event(ret);
    }
    return ret;
}

inline hal::event_ptr
object::acquire(GmacProtection prot, gmacError_t &err)
{
    lock_write();
    hal::event_ptr ret;
    TRACE(LOCAL, "Acquiring object %p?", addr_);
    if (released_) {
        TRACE(LOCAL, "Acquiring object %p", addr_);
        ret = coherence_op<GmacProtection>(&protocol::acquire, prot, err);
        if (err == gmacSuccess) {
            set_last_event(ret);
        }
    }
    released_ = false;
    unlock();
    return ret;
}

inline hal::event_ptr 
object::release(bool flushDirty, gmacError_t &err)
{
    hal::event_ptr ret;
    lock_write();
    TRACE(LOCAL, "Releasing object %p?", addr_);
    if (flushDirty && !released_) {
        TRACE(LOCAL, "Releasing object %p", addr_);
        ret = coherence_op(&protocol::release, err);
    }
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

#ifdef USE_VM
inline gmacError_t
object::acquireWithBitmap()
{
    lock_read();
    gmacError_t ret = coherence_op(&protocol::acquireWithBitmap);
    unlock();
    return ret;
}
#endif

#if 0
template <typename P1, typename P2>
gmacError_t
object::for_each_block(gmacError_t (protocol::*op)(block_ptr, P1 &, P2), P1 &p1, P2 p2)
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
#endif

inline hal::event_ptr 
object::to_host(gmacError_t &err)
{
    lock_read();
    hal::event_ptr ret = coherence_op(&protocol::to_host, err);
    if (err == gmacSuccess) {
        set_last_event(ret);
    }
    unlock();
    return ret;
}

inline hal::event_ptr
object::to_device(gmacError_t &err)
{
    lock_read();
    hal::event_ptr ret = coherence_op(&protocol::release, err);
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
    gmacError_t ret = memoryOp(&protocol::copyToBuffer, buffer, size,
                               bufferOffset, objectOffset);
    unlock();
    return ret;
}

inline gmacError_t object::copyFromBuffer(core::io_buffer &buffer, size_t size,
                                          size_t bufferOffset, size_t objectOffset)
{
    lock_read();
    gmacError_t ret = memoryOp(&protocol::copyFromBuffer, buffer, size,
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
        gmacError_t ret = memoryOp(&protocol::copyFromBuffer, buffer, size,
        bufferOffset, objectOffset);
    dst.unlock();
    src.unlock();
    return ret;
}
#endif

}}

#endif
