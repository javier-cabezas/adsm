#include "types.h"

namespace __impl { namespace hal { namespace cuda {

operation::state
operation::get_state()
{
    stream_->get_aspace().set();

    CUresult res = cuEventQuery(eventEnd_);

    // Advance untile we find a not ready event
    if (res == CUDA_ERROR_NOT_READY) {
        return End;
    } else {
        return Start;
    }
}

#if 0
event::state
event::get_state()
{
    if (state_ != state::End) {
        if (operations_.size() == 0) {
            // No operations enqueued, we are done
            state_ = state::End;
        } else {
            if (syncOpBegin_ == operations_.end()) {
                // No operations to sync, we are done
                state_ = state::End;
            } else {
                while (syncOpBegin_ != operations_.end()) {
                    state_ = (*syncOpBegin_)->get_state();

                    // Advance untile we find a not ready event
                    if (state_ != state::End) {
                        break;
                    } else {
                        ++syncOpBegin_;
                    }
                }
            }
        }
    }

    return state_;
}

void
_event_common_t::set_barrier(virt::aspace &as, CUstream s)
{
    //stream &s = reinterpret_cast<stream &>(_stream);
    as.set();

    for (auto &op : operations_) {
        CUresult res = cuStreamWaitEvent(s, op.eventEnd_, 0);
        ASSERTION(res == CUDA_SUCCESS, "Error adding barrier");
    }
}

event::state
_event_common_t::get_state()
{
    if (state_ != End) {
        if (operations_.size() == 0) {
            // No operations enqueued, we are done
            state_ = End;
        } else {
            if (syncOpBegin_ == operations_.end()) {
                // No operations to sync, we are done
                state_ = End;
            } else {
                while (syncOpBegin_ != operations_.end()) {
                    syncOpBegin_->stream_.get_aspace().set();

                    CUresult res = cuEventQuery(syncOpBegin_->eventEnd_);

                    // Advance untile we find a not ready event
                    if (res == CUDA_ERROR_NOT_READY) {
                        break;
                    } else {
                        ++syncOpBegin_;
                    }
                }

                if (syncOpBegin_ == operations_.end()) {
                    // All queued operations are finished, we are done
                    state_ = End;
                } else {
                    // Some remaining operations
                    state_ = Queued;
                }
            }
        }
    }

    return state_;
}
#endif


void
operation::reset(type t, bool async, stream &s)
{
    async_ = async;
    type_ = t;
    err_ = hal::error::HAL_SUCCESS;
    synced_ = false;
    stream_ = &s;
}

hal::error
list_event::sync()
{
    hal::error ret = hal::error::HAL_SUCCESS;
    for (parent::const_iterator it  = parent::begin();
                                it != parent::end();
                              ++it) {
        ret = (*it)->sync();
        if (ret != hal::error::HAL_SUCCESS) break;
    }

    return ret;
}

#if 0
void
list_event::set_barrier(hal_stream &_stream)
{
    stream &s = reinterpret_cast<stream &>(_stream);
    s.get_aspace().set();

    for (parent::iterator it  = parent::begin();
                          it != parent::end();
                        ++it) {
        (**it).set_synced();
    }
}
#endif

void
list_event::set_synced()
{
    for (parent::const_iterator it  = parent::begin();
                                it != parent::end();
                              ++it) {
        (**it).set_synced();
    }
}


}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
