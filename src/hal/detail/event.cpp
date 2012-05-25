#include "event.h"

namespace __impl { namespace hal { namespace detail {

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



hal::error
event::sync()
{
    lock_write();

    if (synced_ == false) {
        TRACE(LOCAL, FMT_ID2" sync", get_print_id2());

        while (syncOpBegin_ != operations_.end()) {
            err_ = (*syncOpBegin_)->sync();
            if (err_ != HAL_SUCCESS) {
                unlock();
                return err_;
            }
            ++syncOpBegin_;
        }
        synced_ = true;
    }

    unlock();

    if (err_ == HAL_SUCCESS) {
        // Execute pending operations associated to the event
        exec_triggers(true);
    }

    return err_;
}

void
event::set_synced()
{
    lock_write();

    if (synced_ == false) {
        TRACE(LOCAL, FMT_ID2" setting event as synced", get_print_id2());
#ifdef USE_TRACE
        get_stream().get_context().set();

        float mili;
        cl_int res;
        res = cuEventElapsedTime(&mili, eventStart_, eventEnd_);
        if (res == CUDA_SUCCESS) {
            timeQueued_ = timeSubmit_ = timeStart_ = timeBase_;
            timeEnd_ = timeQueued_ + time_t(mili * 1000.f);
        }
        err_ = error(res);
#endif
        // Execute pending operations associated to the event
        exec_triggers(true);
        synced_ = true;
    }

    unlock();
}


}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
