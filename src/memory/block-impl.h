#ifndef GMAC_MEMORY_BLOCK_IMPL_H_
#define GMAC_MEMORY_BLOCK_IMPL_H_

#include "Memory.h"
#ifdef USE_VM
#include "vm/Bitmap.h"
#include "core/Mode.h"
#endif

namespace __impl { namespace memory {

inline
block::block(hostptr_t addr, hostptr_t shadow, size_t size) :
    gmac::util::mutex("Block"),
    size_(size),
    addr_(addr),
    shadow_(shadow)
{
}

inline block::~block()
{
}

inline hostptr_t block::addr() const
{
    return addr_;}

inline size_t block::size() const
{
    return size_;
}

#if 0
inline gmacError_t block::signal_read(hostptr_t addr)
{
    TRACE(LOCAL,"SIGNAL READ on block %p: addr %p", addr_, addr);
    lock();
    gmacError_t ret = protocol_.signal_read(*this, addr);
    unlock();
    return ret;
}

inline gmacError_t block::signal_write(hostptr_t addr)
{
    TRACE(LOCAL,"SIGNAL WRITE on block %p: addr %p", addr_, addr);
    lock();
    gmacError_t ret = protocol_.signal_write(*this, addr);
    unlock();
    return ret;
}

template <typename R>
inline R block::coherenceOp(R (protocol_interface::*f)(block_ptr))
{
    lock();
    R ret = (protocol_.*f)(*this);
    unlock();
    return ret;
}

template <typename R, typename T>
inline R block::coherenceOp(R (protocol_interface::*op)(block_ptr, T &), T &param)
{
    lock();
    R ret = (protocol_.*op)(*this, param);
    unlock();
    return ret;
}

inline
hal::event_t
block::copy_op(protocol_interface::CopyOp1To op, block_ptr dst, size_t dstOff, const hostptr_t src, size_t count, gmacError_t &err)
{
    dst->lock();
    hal::event_t ret = (dst->protocol_.*op)(dst, dstOff, src, count, err);
    dst->unlock();

    return ret;
}

inline
hal::event_t
block::copy_op(protocol_interface::CopyOp1From op, hostptr_t dst, block_ptr src, size_t srcOff, size_t count, gmacError_t &err)
{
    src->lock();
    hal::event_t ret = (src.protocol_.*op)(dst, src, srcOff, count, err);
    src->unlock();

    return ret;
}

inline hal::event_t
block::copy_op(protocol_interface::CopyOp2 op, block_ptr dst, size_t dstOff, block_ptr src, size_t srcOff, size_t count, gmacError_t &err)
{
    src->lock();
    dst->lock();
    hal::event_t ret = (dst.protocol_.*op)(dst, dstOff, src, srcOff, count, err);
    dst->unlock();
    src->unlock();
    return ret;
}

inline hal::event_t
block::device_op(protocol_interface::DeviceOpTo op,
                 hal::device_output &output,
                 block_ptr b, size_t blockOffset,
                 size_t count, gmacError_t &err)
{
    b->lock();
    hal::event_t ret =(b.protocol_.*op)(output, b, blockOffset, count, err);
    b->unlock();
    return ret;
}

inline hal::event_t
block::device_op(protocol_interface::DeviceOpFrom op,
		         block_ptr b, size_t blockOffset,
                 hal::device_input &input,
                 size_t count, gmacError_t &err)
{
    b->lock();
    hal::event_t ret =(b.protocol_.*op)(b, blockOffset, input, count, err);
    b->unlock();
    return ret;
}

inline gmacError_t block::memset(int v, size_t size, size_t blockOffset)
{
    lock();
    gmacError_t ret;
    hal::event_t event = protocol_.memset(*this, blockOffset, v, size, ret);
    unlock();
    return ret;
}
#endif


#if 0
inline gmacError_t
block::dump(std::ostream &param, protocol_interface::common::Statistic stat)
{
    lock();
    gmacError_t ret = protocol_.dump(*this, param, stat);
    unlock();
    return ret;
}
#endif

inline
hostptr_t
block::get_shadow() const
{
    return shadow_;
}

}}

#endif
