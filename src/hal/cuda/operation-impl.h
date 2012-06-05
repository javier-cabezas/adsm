#ifndef GMAC_HAL_CUDA_OPERATION_IMPL_H_
#define GMAC_HAL_CUDA_OPERATION_IMPL_H_

namespace __impl { namespace hal { namespace cuda {

inline
operation::operation(parent::type t, bool async, virt::aspace &as, stream &s) :
    parent(t, async),
    observer_destruct(),
    as_(as),
    stream_(&s),
    aspaceValid_(true)
{
    TRACE(LOCAL, FMT_ID2 " create: %p", get_print_id2(), this);
    CUresult err;
    err = cuEventCreate(&eventStart_, CU_EVENT_DEFAULT);
    ASSERTION(err == CUDA_SUCCESS);
    err = cuEventCreate(&eventEnd_, CU_EVENT_DEFAULT);
    ASSERTION(err == CUDA_SUCCESS);

    static_cast<util::observable<detail::virt::aspace, util::event::destruct> &>(as).add_observer(*this);
}

inline
void
operation::clean_CUDA()
{
    TRACE(LOCAL, FMT_ID2 " clean cuda events", get_print_id2());
    CUresult err;
    if (eventStart_ != nullptr) {
        err = cuEventDestroy(eventStart_);
        ASSERTION(err == CUDA_SUCCESS);
        eventStart_ = nullptr;
    }
    if (eventEnd_ != nullptr) {
        err = cuEventDestroy(eventEnd_);
        ASSERTION(err == CUDA_SUCCESS);
        eventEnd_ = nullptr;
    }
}

inline
operation::~operation()
{
    TRACE(LOCAL, FMT_ID2 " destroy: %p", get_print_id2(), this);
    if (aspaceValid_) {
        static_cast<util::observable<detail::virt::aspace, util::event::destruct> &>(as_).remove_observer(*this);
        clean_CUDA(); 
    }
}

inline void
operation::event_handler(detail::virt::aspace &aspace, const util::event::destruct &)
{
    TRACE(LOCAL, FMT_ID2 " " FMT_ID2 "deleted!", get_print_id2(), aspace.get_print_id2());
    ASSERTION(aspace.get_id() == as_.get_id());
    clean_CUDA(); 
    aspaceValid_ = false;
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
    hal::error ret = hal::error::HAL_SUCCESS;
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

template <class Type>
void delete_cpu(void *p)
{
    delete static_cast<Type *>(p);
}

template <class Type>
void delete_gpu(void *p)
{
    delete static_cast<Type *>(p);
}

inline std::unique_ptr<operation, void(*)(void *)>
create_op(operation::type t, bool async, virt::aspace &as, stream &s)
{
    return std::unique_ptr<operation, void(*)(void *)>(as.get_new_op(t, async, s), delete_gpu<operation>);
}

inline std::unique_ptr<hal::cpu::operation, void(*)(void *)>
create_cpu_op(operation::type t, bool async)
{
    return std::unique_ptr<hal::cpu::operation, void(*)(void *)>(new hal::cpu::operation(t, async), delete_cpu<hal::cpu::operation>);
}


}}}

#endif /* TYPES_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
