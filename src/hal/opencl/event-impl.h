#ifndef GMAC_HAL_OPENCL_EVENT_IMPL_H_
#define GMAC_HAL_OPENCL_EVENT_IMPL_H_

namespace __impl { namespace hal { namespace opencl {

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

    timeBase_ = hal::get_timestamp();
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
        TRACE(LOCAL, "event<"FMT_ID">: waiting for event", get_print_id());
        cl_int res = clWaitForEvents(1, &event_);
        if (res == CL_SUCCESS) {
#ifdef USE_TRACE
            cl_ulong queued;
            cl_ulong submit;
            cl_ulong start;
            cl_ulong end;
            res = clGetEventProfilingInfo(event_,
                                          CL_PROFILING_COMMAND_QUEUED, 
                                          sizeof(cl_ulong), 
                                          &queued,
                                          NULL);
            ASSERTION(res == CL_SUCCESS);
            res = clGetEventProfilingInfo(event_,
                                          CL_PROFILING_COMMAND_SUBMIT, 
                                          sizeof(cl_ulong), 
                                          &submit,
                                          NULL);
            ASSERTION(res == CL_SUCCESS);
            res = clGetEventProfilingInfo(event_,
                                          CL_PROFILING_COMMAND_START, 
                                          sizeof(cl_ulong), 
                                          &start,
                                          NULL);
            ASSERTION(res == CL_SUCCESS);
            res = clGetEventProfilingInfo(event_,
                                          CL_PROFILING_COMMAND_END, 
                                          sizeof(cl_ulong), 
                                          &end,
                                          NULL);
            ASSERTION(res == CL_SUCCESS);

            timeQueued_ = timeBase_;
            timeSubmit_ = timeBase_ + (submit - queued) / 1000;
            timeStart_  = timeBase_ + (start - queued) / 1000;
            timeEnd_    = timeBase_ + (end - queued) / 1000;

#endif
            // Execute pending operations associated to the event
            exec_triggers();
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
        cl_ulong queued;
        cl_ulong submit;
        cl_ulong start;
        cl_ulong end;
        cl_int res;
        res = clGetEventProfilingInfo(event_,
                                      CL_PROFILING_COMMAND_QUEUED, 
                                      sizeof(cl_ulong), 
                                      &queued,
                                      NULL);
        ASSERTION(res == CL_SUCCESS);
        res = clGetEventProfilingInfo(event_,
                                      CL_PROFILING_COMMAND_SUBMIT, 
                                      sizeof(cl_ulong), 
                                      &submit,
                                      NULL);
        ASSERTION(res == CL_SUCCESS);
        res = clGetEventProfilingInfo(event_,
                                      CL_PROFILING_COMMAND_START, 
                                      sizeof(cl_ulong), 
                                      &start,
                                      NULL);
        ASSERTION(res == CL_SUCCESS);
        res = clGetEventProfilingInfo(event_,
                                      CL_PROFILING_COMMAND_END, 
                                      sizeof(cl_ulong), 
                                      &end,
                                      NULL);
        ASSERTION(res == CL_SUCCESS);

        timeQueued_ = timeBase_;
        timeSubmit_ = timeBase_ + (submit - queued) / 1000;
        timeStart_  = timeBase_ + (start - queued) / 1000;
        timeEnd_    = timeBase_ + (end - queued) / 1000;
        err_ = error(res);
#endif
        // Execute pending operations associated to the event
        exec_triggers();
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

inline
event_ptr::event_ptr()
{
}

#ifdef USE_CXX0X
inline
event_ptr::event_ptr(event_ptr &&event) :
    ptrEvent_(std::move(event.ptrEvent_))
{
}
#endif

inline
event_ptr::event_ptr(const event_ptr &event) :
    ptrEvent_(event.ptrEvent_)
{
}

inline
event_ptr &
event_ptr::operator=(const event_ptr &event)
{
    if (&event != this) {
        ptrEvent_ = event.ptrEvent_;
    }

    return *this;
}

#ifdef USE_CXX0X
inline
event_ptr &
event_ptr::operator=(event_ptr &&event)
{
    if (&event != this) {
        ptrEvent_ = std::move(event.ptrEvent_);
    }

    return *this;
}
#endif

template <typename F>
inline
void
event_ptr::add_trigger(F fun)
{
    ASSERTION(bool(ptrEvent_));

    ptrEvent_->add_trigger(fun);
}

}}}

#endif /* GMAC_HAL_OPENCL_EVENT_IMPL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
