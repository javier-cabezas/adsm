#ifndef GMAC_CORE_VDEVICE_IMPL_H_
#define GMAC_CORE_VDEVICE_IMPL_H_

#include "config/order.h"
#include "memory/object.h"
#include "memory/protocol.h"
#include "trace/Tracer.h"

namespace __impl { namespace core {

inline
vdevice::vdevice() :
    util::Reference("vdevice")
{
    TRACE(LOCAL,"Creating Execution vdevice %p", this);
    trace::StartThread(THREAD_T(get_print_id()), "GPU");
    SetThreadState(THREAD_T(get_print_id()), trace::Idle);
}

inline
vdevice::~vdevice()
{
    trace::EndThread(THREAD_T(get_print_id()));
    TRACE(LOCAL,"Destroying Execution vdevice %p", this);
}


} }

#endif
