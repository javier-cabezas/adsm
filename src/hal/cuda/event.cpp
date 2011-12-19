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
        get_stream().get_context().set();

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
    for (Parent::iterator it  = Parent::begin();
            it != Parent::end();
            it++) {
        ret = (*it)->sync();
        if (ret != gmacSuccess) break;
    }

    return ret;
}

void
list_event::add_event(event_ptr event) 
{
    Parent::push_back(event);
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
