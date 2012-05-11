#ifndef GMAC_HAL_CUDA_EVENT_IMPL_H_
#define GMAC_HAL_CUDA_EVENT_IMPL_H_

namespace __impl { namespace hal { namespace cuda {

inline
operation::operation(parent::type t, bool async, stream &s) :
    parent(t, async),
    stream_(s)
{
    CUresult err;
    err = cuEventCreate(&eventStart_, CU_EVENT_DEFAULT);
    ASSERTION(err == CUDA_SUCCESS);
    err = cuEventCreate(&eventEnd_, CU_EVENT_DEFAULT);
    ASSERTION(err == CUDA_SUCCESS);
}

inline
operation::func_op::result_type
operation::execute(func_op f)
{
    stream_.get_aspace().set(); 

    CUresult err = cuEventRecord(eventStart_, stream_());
    ASSERTION(err == CUDA_SUCCESS);
    func_op::result_type ret = f(stream_());
    err = cuEventRecord(eventEnd_, stream_());
    ASSERTION(err == CUDA_SUCCESS);

    // Modify the state since there is a new operation
    state_ = Queued;

    return ret;
}

inline
gmacError_t
operation::sync()
{
    gmacError_t ret = gmacSuccess;
    if (synced_ == false) {
        stream_.get_aspace().set();

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
            synced_ = true;
        }
        ret = error(res);
    }
    return ret;
}

inline
void
operation::set_barrier(hal::detail::stream &_s)
{
    stream &s = reinterpret_cast<stream &>(_s);
    CUresult res = cuStreamWaitEvent(s(), eventEnd_, 0);
    ASSERTION(res == CUDA_SUCCESS, "Error adding barrier");
}

inline
gmacError_t
_event_t::sync()
{
    lock_write();

    if (synced_ == false) {
        TRACE(LOCAL, FMT_ID2" sync", get_print_id2());

        while (syncOpBegin_ != operations_.end()) {
            err_ = (*syncOpBegin_)->sync();
            if (err_ != gmacSuccess) {
                unlock();
                return err_;
            }
            ++syncOpBegin_;
        }
        synced_ = true;
    }

    unlock();

    if (err_ == gmacSuccess) {
        // Execute pending operations associated to the event
        exec_triggers(true);
    }

    return err_;

}

#if 0
inline
_event_common_t::_event_common_t(bool async, parent::type t, virt::aspace &as) :
    parent(async, t, as),
    syncOpBegin_(operations_.end())
{
}

template <typename R>
inline
R
_event_common_t::add_operation(hal_event_ptr ptr, stream &stream, std::function<R()> f)
{
    // During operation execution, the event is not thread-safe
    if (operations_.size() == 0) {
#ifdef USE_TRACE
        // Get the base time if this is the first operation of the event
        timeBase_ = hal::get_timestamp();
#endif
    }
    operation op(stream);

    R r = op.execute(f);

    operations_.push_back(op);

    // Compute the first operation to be synchronized
    if (operations_.size() == 1) {
        syncOpBegin_ = operations_.begin();
    } else if (syncOpBegin_ == operations_.end()) {
        syncOpBegin_ = std::prev(operations_.end());
    }

    // Modify the state since there is a new operation
    state_ = Queued;

    stream.set_last_event(ptr);

    return r;
}

inline
gmacError_t
_event_common_t::sync_no_exec()
{
    lock_write();

    if (synced_ == false) {
        TRACE(LOCAL, FMT_ID2" sync", get_print_id2());

        while (syncOpBegin_ != operations_.end()) {
            err_ = syncOpBegin_->sync();
            if (err_ != gmacSuccess) {
                unlock();
                return err_;
            }
            ++syncOpBegin_;
        }
        synced_ = true;
    }

    unlock();

    return err_;
}

inline
gmacError_t
_event_common_t::sync()
{
    sync_no_exec();

    if (err_ == gmacSuccess) {
        // Execute pending operations associated to the event
        exec_triggers(true);
    }

    return err_;
}
#endif



#if 0
inline
stream &
_event_common_t::get_stream()
{
    ASSERTION(stream_ != NULL);

    return *stream_;
}
#endif

inline
typename operation::func_op::result_type
_event_t::add_operation(hal_event_ptr ptr, stream &stream, operation::func_op f, operation::type t, bool async)
{
    // During operation execution, the event is not thread-safe
    if (operations_.size() == 0) {
#ifdef USE_TRACE
        // Get the base time if this is the first operation of the event
        timeBase_ = hal::get_timestamp();
#endif
    }
    operation *op = new operation(t, async, stream);

    operation::func_op::result_type r = op->execute(f);

    operations_.push_back(op);

    // Compute the first operation to be synchronized
    if (operations_.size() == 1) {
        syncOpBegin_ = operations_.begin();
    } else if (syncOpBegin_ == operations_.end()) {
        syncOpBegin_ = std::prev(operations_.end());
    }

    stream.set_last_event(ptr);

    return r;
}



inline
_event_t::_event_t(bool async, type t, virt::aspace &as) :
    parent(async, t),
    as_(as)
{
}

inline
virt::aspace &
_event_t::get_vaspace()
{
    return as_;
}

inline
void
_event_t::set_synced()
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

inline
void
event_deleter::operator()(_event_t *ev)
{
    ev->get_vaspace().dispose_event(*ev);
}

#if 0

inline
event_ptr::event_ptr(bool async, typename _event_t::type t, aspace &context) :
    ptrEvent_(context.get_new_event(async, t), event_deleter())
{
}
#endif


inline static
event_ptr
create_event(bool async, _event_t::type t, virt::aspace &as)
{
    return event_ptr(as.get_new_event(async, t), event_deleter());
}

}}}

#endif /* TYPES_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
