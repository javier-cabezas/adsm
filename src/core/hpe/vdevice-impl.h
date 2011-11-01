#ifndef GMAC_CORE_HPE_VDEVICE_IMPL_H_
#define GMAC_CORE_HPE_VDEVICE_IMPL_H_

#include "memory/object.h"

#include "core/hpe/process.h"
#include "core/hpe/context.h"
#include "core/io_buffer.h"

namespace __impl { namespace core { namespace hpe {

#ifdef USE_VM
inline memory::vm::Bitmap &
vdevice::getDirtyBitmap()
{
    return bitmap_;
}

inline const memory::vm::Bitmap &
vdevice::getDirtyBitmap() const
{
    return bitmap_;
}
#endif


inline process &
vdevice::get_process()
{
    return proc_;
}

inline const process &
vdevice::get_process() const
{
    return proc_;
}

#if 0
inline
gmacError_t
vdevice::prepareForCall()
{
    trace::SetThreadState(trace::Wait);
    gmacError_t ret = streamToAccelerator_.sync();
    if (ret == gmacSuccess) ret = streamToHost_.sync();
    trace::SetThreadState(trace::Idle);
    return ret;
}

inline gmacError_t
vdevice::wait(core::hpe::KernelLaunch &launch)
{
    // TODO: use an event for this
    gmacError_t ret = streamLaunch_->sync();

    return ret;
}
#endif

inline gmacError_t
vdevice::wait()
{
    gmacError_t ret = streamLaunch_.sync();

    return ret;
}

inline hal::stream_t&
vdevice::eventStream()
{
    return streamLaunch_;
}

inline
util::smart_ptr<address_space>::shared
vdevice::get_address_space()
{
    return aspace_;
}

inline
util::smart_ptr<const address_space>::shared
vdevice::get_address_space() const
{
    return aspace_;
}

}}}

#endif
