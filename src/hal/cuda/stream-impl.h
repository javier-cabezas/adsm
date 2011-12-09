#ifndef GMAC_HAL_CUDA_STREAM_IMPL_H_
#define GMAC_HAL_CUDA_STREAM_IMPL_H_

namespace __impl { namespace hal { namespace cuda {

inline
stream_t::Parent::state
stream_t::query()
{
    Parent::state ret = Running;

    get_context().set();

    CUresult res = cuStreamQuery(stream_);

    if (res == CUDA_ERROR_NOT_READY) {
        ret = Running;
    } else if (res == CUDA_SUCCESS) {
        ret = Empty;
    } else {
        FATAL("Unhandled case");
    }

    return ret;
}

inline
void
stream_t::set_last_event(event_t event)
{
    lastEvent_ = event;
}

inline
event_t
stream_t::get_last_event()
{
    return lastEvent_;
}

inline
context_t &
stream_t::get_context()
{
    return reinterpret_cast<context_t &>(Parent::get_context());
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
