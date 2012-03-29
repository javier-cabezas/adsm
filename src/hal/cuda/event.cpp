#include "types.h"

namespace __impl { namespace hal { namespace cuda {

void
_event_common_t::set_barrier(hal_stream &_stream)
{
    stream &s = reinterpret_cast<stream &>(_stream);
    s.get_aspace().set();

    for (auto &op : operations_) {
        CUresult res = cuStreamWaitEvent(s(), op.eventEnd_, 0);
        ASSERTION(res == CUDA_SUCCESS, "Error adding barrier");
    }
}

_event_t::state
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


void
_event_t::reset(bool async, type t)
{
    async_ = async;
    type_ = t;
    err_ = gmacSuccess;
    synced_ = false;
    state_ = None;

    remove_triggers();
}

gmacError_t
list_event::sync()
{
    gmacError_t ret = gmacSuccess;
    for (parent::iterator it  = parent::begin();
                          it != parent::end();
                        ++it) {
        ret = (*it)->sync();
        if (ret != gmacSuccess) break;
    }

    return ret;
}

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

void
list_event::set_synced()
{
    for (parent::iterator it  = parent::begin();
                          it != parent::end();
                        ++it) {
        (**it).set_synced();
    }
}

void
list_event::add_event(hal_event_ptr event) 
{
    parent::push_back(util::reinterpret_ptr<_event_t, virt::hal_event>(event));
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
