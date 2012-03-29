#ifndef GMAC_HAL_CUDA_STREAM_IMPL_H_
#define GMAC_HAL_CUDA_STREAM_IMPL_H_

namespace __impl { namespace hal { namespace cuda {

inline
stream::stream(virt::aspace &as) :
    parent(as),
    valid_(false)
{
}

inline
stream::stream(CUstream stream, virt::aspace &as) :
    parent(as),
    valid_(true),
    stream_(stream)
{
    TRACE(LOCAL, "Creating stream: %p", (*this)());
}

inline
void
stream::set_cuda_stream(CUstream s)
{
    stream_ = s;
    valid_ = true;
}

inline
gmacError_t
stream::sync()
{
    ASSERTION(valid_);

    get_aspace().set(); 

    TRACE(LOCAL, "Waiting for stream: %p", (*this)());
    CUresult ret = cuStreamSynchronize((*this)());

    return cuda::error(ret);
}

inline
stream::parent::state
stream::query()
{
    ASSERTION(valid_);

    parent::state ret = Running;

    get_aspace().set();

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
virt::aspace &
stream::get_aspace()
{
    ASSERTION(valid_);

    return reinterpret_cast<virt::aspace &>(parent::get_aspace());
}

inline
CUstream &
stream::operator()()
{
    ASSERTION(valid_);

    return stream_;
}

inline
const CUstream &
stream::operator()() const
{
    ASSERTION(valid_);

    return stream_;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
