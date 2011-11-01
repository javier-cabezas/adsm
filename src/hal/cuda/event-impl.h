#ifndef GMAC_HAL_CUDA_EVENT_IMPL_H_
#define GMAC_HAL_CUDA_EVENT_IMPL_H_

namespace __impl { namespace hal { namespace cuda {

inline
_event_common_t::_event_common_t() :
    stream_(NULL)
{
}

inline
void
_event_common_t::begin(stream_t &stream)
{
    stream_ = &stream;

    stream.get_context().set(); 

    CUresult err;
    err = cuEventCreate(&eventStart_, CU_EVENT_DEFAULT);
    ASSERTION(err == CUDA_SUCCESS);
    err = cuEventCreate(&eventEnd_, CU_EVENT_DEFAULT);
    ASSERTION(err == CUDA_SUCCESS);

    timeBase_ = hal::get_timestamp();
    err = cuEventRecord(eventStart_, stream());
    ASSERTION(err == CUDA_SUCCESS);
}

inline
void
_event_common_t::end()
{
    ASSERTION(stream_ != NULL);
    stream_->get_context().set(); 

    cuEventRecord(eventEnd_, (*stream_)());
}

inline
stream_t &
_event_common_t::get_stream()
{
    ASSERTION(stream_ != NULL);

    return *stream_;
}

inline
event_t::event_t(Parent::type t, context_t &context) :
    Parent(t, context)
{
}

inline
async_event_t::async_event_t(event_t::type t, context_t &context) :
    Parent(),
    event_t(t, context)
{
}

inline
gmacError_t
async_event_t::sync()
{
    gmacError_t ret;
    if (Parent::is_synced() == false) {
        get_stream().get_context().set();

        TRACE(LOCAL, "Waiting for event: %p", eventEnd_);
        CUresult res = cuEventSynchronize(eventEnd_);
        if (res == CUDA_SUCCESS) {
            float mili;
            res = cuEventElapsedTime(&mili, eventStart_, eventEnd_);
            if (res == CUDA_SUCCESS) {
                timeQueued_ = timeSubmit_ = timeStart_ = timeBase_;
                timeEnd_ = timeQueued_ + time_t(mili * 1000.f);
            }
        }
        set_synced(true);
        ret = error(res);
    } else {
        ret = err_;
    }

    return ret;
}

}}}

#endif /* TYPES_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
