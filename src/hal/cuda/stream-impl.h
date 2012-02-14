#ifndef GMAC_HAL_CUDA_STREAM_IMPL_H_
#define GMAC_HAL_CUDA_STREAM_IMPL_H_

namespace __impl { namespace hal { namespace cuda {

inline
stream::stream(CUstream stream, aspace &context) :
    parent(context),
    stream_(stream)
{
    TRACE(LOCAL, "Creating stream: %p", (*this)());
}

inline
gmacError_t
stream::sync()
{
    get_aspace().set(); 

    TRACE(LOCAL, "Waiting for stream: %p", (*this)());
    CUresult ret = cuStreamSynchronize((*this)());

    return cuda::error(ret);
}

inline
stream::parent::state
stream::query()
{
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
aspace &
stream::get_aspace()
{
    return reinterpret_cast<aspace &>(parent::get_aspace());
}

inline
CUstream &
stream::operator()()
{
    return stream_;
}

inline
const CUstream &
stream::operator()() const
{
    return stream_;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
