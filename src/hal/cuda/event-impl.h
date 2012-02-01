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
_event_t::_event_t(bool async, type t, context_t &context) :
    Parent(async, t, context)
{
}

inline
gmacError_t
_event_t::sync()
{
    gmacError_t ret;
    lock_write();

    if (synced_ == false) {
        get_stream().get_context().set();

        TRACE(LOCAL, "event<"FMT_ID">: waiting for event", get_print_id());
        CUresult res = cuEventSynchronize(eventEnd_);
        if (res == CUDA_SUCCESS) {
#ifdef USE_TRACE
            float mili;
            res = cuEventElapsedTime(&mili, eventStart_, eventEnd_);
            if (res == CUDA_SUCCESS) {
                timeQueued_ = timeSubmit_ = timeStart_ = timeBase_;
                timeEnd_ = timeQueued_ + time_t(mili * 1000.f);
            }
#endif
            // Execute pending operations associated to the event
            exec_triggers(true);
            synced_ = true;
        }
        err_ = error(res);
    }

    ret = err_;

    unlock();

    return ret;
}

inline
void
_event_t::set_synced()
{
    lock_write();

    if (synced_ == false) {
        TRACE(LOCAL, "event<"FMT_ID">: setting event as synced", get_print_id());
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

inline
void
event_deleter::operator()(_event_t *ev)
{
    ev->get_context().dispose_event(*ev);
}

inline
event_ptr::event_ptr(bool async, _event_t::type t, context_t &context) :
    ptrEvent_(context.get_new_event(async, t), event_deleter())
{
}

}}}

#endif /* TYPES_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
