#ifndef GMAC_HAL_CUDA_STREAM_IMPL_H_
#define GMAC_HAL_CUDA_STREAM_IMPL_H_

namespace __impl { namespace hal { namespace cuda {

inline
stream_t::stream_t(CUstream stream, aspace &context) :
    Parent(context),
    stream_(stream)
{
    TRACE(LOCAL, "Creating stream: %p", (*this)());
}

inline
gmacError_t
stream_t::sync()
{
    get_context().set(); 

    TRACE(LOCAL, "Waiting for stream: %p", (*this)());
    CUresult ret = cuStreamSynchronize((*this)());

    return cuda::error(ret);
}

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
aspace &
stream_t::get_context()
{
    return reinterpret_cast<aspace &>(Parent::get_context());
}

inline
CUstream &
stream_t::operator()()
{
    return stream_;
}

inline
const CUstream &
stream_t::operator()() const
{
    return stream_;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
