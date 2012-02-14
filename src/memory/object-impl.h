#ifndef GMAC_MEMORY_OBJECT_IMPL_H_
#define GMAC_MEMORY_OBJECT_IMPL_H_

#include <functional>

#include <fstream>
#include <sstream>

#include "memory.h"
#include "protocol.h"

#include "trace/logger.h"

namespace __impl { namespace memory {

#define protocol_member(f,...) util::bind(std::mem_fn(&f), &protocol_, std::placeholders::_1, __VA_ARGS__)

inline object::object(protocol &protocol, host_ptr addr, size_t size,
                      int flagsHost, int flagsDevice) :
    lock("object"),
    protocol_(protocol),
    addr_(addr),
    shadow_(NULL),
    size_(size),
    flagsHost_(flagsHost),
    flagsDevice_(flagsDevice)
{
}

#ifdef DEBUG
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

    lock::lock();
    switch (event->get_type()) {
    case hal::event_type::TransferToHost:
        lastToHost_ = event;
        break;
    case hal::event_type::TransferToDevice:
        lastToDevice_ = event;
        break;
    case hal::event_type::TransferHost:
        lastHost_ = event;
        break;
    case hal::event_type::TransferDevice:
        lastDevice_ = event;
        break;
    case hal::event_type::Kernel:
        lastKernel_ = event;
        break;
    case hal::event_type::Invalid:
        goto no_update_last_event;
    }

    last_ = event;
no_update_last_event:
    lock::unlock();
}

inline hal::event_ptr
object::get_last_event(hal::event_type type) const
{
    lock::lock();
    switch (type) {
    case hal::event_type::TransferToHost:
        return lastToHost_;
    case hal::event_type::TransferToDevice:
        return lastToDevice_;
    case hal::event_type::TransferHost:
        return lastHost_;
    case hal::event_type::TransferDevice:
        return lastDevice_;
    case hal::event_type::Kernel:
        return lastKernel_;
    case hal::event_type::Invalid:
        return hal::event_ptr();
    }
    lock::unlock();
}

inline
protocol &
object::get_protocol()
{
    return protocol_;
}

inline const object::bounds
object::get_bounds() const
{
    // No need for lock -- addr_ is never modified
    return bounds(addr_, addr_ + size_);
}

inline const object::bounds
object::get_bounds_shadow() const
{
    ASSERTION(shadow_, "Object does not have a shadow copy");
    // No need for lock -- addr_ is never modified
    return bounds(shadow_, shadow_ + size_);
}

inline ssize_t
object::get_block_base(size_t offset) const
{
    return -1 * (offset % get_block_size());
}

inline size_t
object::get_block_end(size_t offset) const
{
    if (offset + get_block_base(offset) + get_block_size() > size_)
        return size_ - offset;
    else
        return size_t(ssize_t(get_block_size()) + get_block_base(offset));
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

inline int
object::get_flags_host() const
{
    return flagsHost_;
}

inline int
object::get_flags_device() const
{
    return flagsDevice_;
}

template <typename F>
hal::event_ptr
object::coherence_op(F f, gmacError_t &err)
{
    hal::event_ptr ret;
    err = gmacSuccess;
    for (const_locking_iterator i  = begin();
                                i != end();
                              ++i) {
        ret = f(*i);
        if (err != gmacSuccess) break;
        set_last_event(ret);
    }
    return ret;
}

inline hal::event_ptr
object::coherence_op(hal::event_ptr (protocol::*f)(protocols::common::block_ptr, gmacError_t &), gmacError_t &err)
{
    hal::event_ptr ret;
    for (const_locking_iterator i  = begin();
                                i != end();
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
    hal::event_ptr ret;
    TRACE(LOCAL, "Acquiring object %p", addr_);
    ret = coherence_op(protocol_member(protocol::acquire, prot, err), err);
    if (err == gmacSuccess) {
        set_last_event(ret);
    }
    return ret;
}

inline hal::event_ptr 
object::release(bool flushDirty, gmacError_t &err)
{
    hal::event_ptr ret;
    TRACE(LOCAL, "Releasing object %p?", addr_);
    if (flushDirty) {
        TRACE(LOCAL, "Releasing object %p", addr_);
        ret = coherence_op(&protocol::release, err);
        if (err == gmacSuccess) {
            set_last_event(ret);
        }
    }
    err = gmacSuccess;
    return ret;
}

#if 0
inline bool
object::is_released() const
{
    bool ret;
    ret = released_;
    return ret;
}
#endif

#ifdef USE_VM
inline gmacError_t
object::acquireWithBitmap()
{
    gmacError_t ret = coherence_op(&protocol::acquireWithBitmap);
    return ret;
}
#endif

inline hal::event_ptr 
object::to_host(gmacError_t &err)
{
    hal::event_ptr ret = coherence_op(&protocol::to_host, err);
    if (err == gmacSuccess) {
        set_last_event(ret);
    }
    return ret;
}

inline hal::event_ptr
object::to_device(gmacError_t &err)
{
    hal::event_ptr ret = coherence_op(&protocol::release, err);
    if (err == gmacSuccess) {
        set_last_event(ret);
    }
    return ret;
}

}}

#endif
