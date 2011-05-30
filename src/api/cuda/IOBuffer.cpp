#include "IOBuffer.h"
#include "Mode.h"

#include "hpe/Mode.h"

namespace __impl { namespace cuda {

#ifdef EXPERIMENTAL
void
IOBuffer::map()
{
    dynamic_cast<hpe::Mode *>(mode_)->getAccelerator().registerMem(registerBase_, registerSize_);
}

void
IOBuffer::unmap()
{
    dynamic_cast<hpe::Mode *>(mode_)->getAccelerator().unregisterMem(registerBase_);
}
#endif

gmacError_t
IOBuffer::wait(bool fromCUDA)
{
    EventMap::iterator it;
    it = map_.find(mode_);
    ASSERTION(state_ == Idle || it != map_.end());

    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ASSERTION(mode_ != NULL);
        CUevent start = it->second.first;
        CUevent end   = it->second.second;
        trace::SetThreadState(trace::Wait);
        ret = mode_->waitForEvent(end, fromCUDA);
        trace::SetThreadState(trace::Running);
        if(state_ == ToHost) DataCommToHost(*mode_, start, end, size_);
        else if(state_ == ToAccelerator) DataCommToAccelerator(*mode_, start, end, size_);
        TRACE(LOCAL,"Buffer %p goes Idle", this);
        state_ = Idle;
#ifdef EXPERIMENTAL
        unmap();
#endif
        mode_  = NULL;
    } else {
        ASSERTION(mode_ == NULL);
    }

    return ret;
}


}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
