#ifndef GMAC_CORE_MODE_IMPL_H_
#define GMAC_CORE_MODE_IMPL_H_

#include "config/order.h"
#include "memory/Object.h"
#include "memory/ObjectMap.h"
#include "memory/Protocol.h"
#include "trace/Tracer.h"

namespace __impl { namespace core {

inline
Mode::Mode() :
    util::Reference("Mode"),
    gmac::util::SpinLock("Mode")
#ifdef USE_VM
    , bitmap_(*this)
#endif
{
    TRACE(LOCAL,"Creating Execution Mode %p", this);
    trace::StartThread(THREAD_T(Unique::getId()), "GPU");
    SetThreadState(THREAD_T(Unique::getId()), trace::Idle);
    protocol_ = memory::ProtocolInit(0);
}

inline
Mode::~Mode()
{
    delete protocol_;
    trace::EndThread(THREAD_T(Unique::getId()));
    TRACE(LOCAL,"Destroying Execution Mode %p", this);
}

inline
memory::Protocol &Mode::getProtocol()
{
    return *protocol_;
}

#ifdef USE_VM
inline memory::vm::Bitmap&
Mode::getBitmap()
{
    return bitmap_;
}

inline const memory::vm::Bitmap&
Mode::getBitmap() const
{
    return bitmap_;
}
#endif


} }

#endif
