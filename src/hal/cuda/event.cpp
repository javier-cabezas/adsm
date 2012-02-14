#include "types.h"

namespace __impl { namespace hal { namespace cuda {

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

_event_t::state
_event_t::get_state()
{
    if (state_ != End) {
        get_stream().get_aspace().set();

        CUresult res = cuEventQuery(eventEnd_);

        if (res == CUDA_ERROR_NOT_READY) {
            state_ = Queued;
        } else if (res == CUDA_SUCCESS) {
            state_ = End;
        }
    }

    return state_;
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
        CUresult res = cuStreamWaitEvent(s(), (*it)->eventEnd_, 0);
        ASSERTION(res == CUDA_SUCCESS, "Error adding barrier");
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
    parent::push_back(util::reinterpret_ptr<_event_t, hal_event>(event));
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
