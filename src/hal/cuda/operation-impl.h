#ifndef GMAC_HAL_CUDA_OPERATION_IMPL_H_
#define GMAC_HAL_CUDA_OPERATION_IMPL_H_

namespace __impl { namespace hal { namespace cuda {

inline
operation::operation(parent::type t, bool async, virt::aspace &as, stream &s) :
    parent(t, async),
    as_(as),
    stream_(&s)
{
    TRACE(LOCAL, "Creating operation");
    CUresult err;
    err = cuEventCreate(&eventStart_, CU_EVENT_DEFAULT);
    ASSERTION(err == CUDA_SUCCESS);
    err = cuEventCreate(&eventEnd_, CU_EVENT_DEFAULT);
    ASSERTION(err == CUDA_SUCCESS);
}

inline
operation::~operation()
{
    TRACE(LOCAL, "Deleting operation");
    CUresult err;
    err = cuEventDestroy(eventStart_);
    ASSERTION(err == CUDA_SUCCESS);
    err = cuEventDestroy(eventEnd_);
    ASSERTION(err == CUDA_SUCCESS);
}

template <typename Func, typename... Args>
inline
auto
operation::execute(Func f, Args... args) -> decltype(f(CUstream(), args...))
{
    as_.set();

    CUresult err = cuEventRecord(eventStart_, (*stream_)());
    ASSERTION(err == CUDA_SUCCESS);
    auto ret = f((*stream_)(), args...);
    err = cuEventRecord(eventEnd_, (*stream_)());
    ASSERTION(err == CUDA_SUCCESS);

    // Modify the state since there is a new operation
    state_ = Queued;

    return ret;
}

inline
hal::error
operation::sync()
{
    hal::error ret = HAL_SUCCESS;
    if (synced_ == false) {
        stream_->get_aspace().set();

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
        ret = error_to_hal(res);
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


#if 0
inline
event::event(type t) :
    parent(t)
{
}
#endif

inline operation *
create_op(operation::type t, bool async, virt::aspace &as, stream &s)
{
    return as.get_new_op(t, async, s);
}

inline hal::cpu::operation *
create_cpu_op(operation::type t, bool async)
{
    return new hal::cpu::operation(t, async);
}


}}}

#endif /* TYPES_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
