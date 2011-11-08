#ifndef GMAC_HAL_CUDA_KERNEL_IMPL_H_
#define GMAC_HAL_CUDA_KERNEL_IMPL_H_

namespace __impl { namespace hal { namespace cuda {

inline
kernel_t::kernel_t(CUfunction func, const std::string &name) :
    Parent(func, name)
{
}

inline 
kernel_t::launch &
kernel_t::launch_config(Parent::config &conf, stream_t &stream)
{
    return *(new launch(*this, conf, stream));
}

inline
kernel_t::launch::launch(kernel_t &parent, Parent::config &conf, stream_t &stream) :
    hal::detail::kernel_t<backend_traits, implementation_traits>::launch(parent, dynamic_cast<config &>(conf), stream)
{
}

inline
event_t
kernel_t::launch::execute(list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = execute(err);
    }

    return ret;
}

inline
event_t
kernel_t::launch::execute(event_t event, gmacError_t &err)
{
    event_t ret;
    err = event.sync();

    if (err == gmacSuccess) {
        ret = execute(err);
    }

    return ret;
}

inline
event_t
kernel_t::launch::execute(gmacError_t &err)
{
    get_stream().get_context().set();

    CUresult res;

    dim3 dimsGlobal = get_config().get_dims_global();
    dim3 dimsGroup = get_config().get_dims_group();

    TRACE(LOCAL, "kernel launch on stream: %p", get_stream()());
    event_t ret(true, _event_t::Kernel, get_stream().get_context());

    ret.begin(get_stream());
    res = cuLaunchKernel(get_kernel()(), dimsGlobal.x,
                                         dimsGlobal.y,
                                         dimsGlobal.z,
                                         dimsGroup.x,
                                         dimsGroup.y,
                                         dimsGroup.z,
                                         get_config().memShared_,
                                         get_stream()(),
                                         (void **) get_config().params_,
                                         NULL);
    ret.end();
    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    }

    return ret;
}



inline
kernel_t::config::config(dim3 global, dim3 group, size_t shared, cudaStream_t tokens) :
    hal::detail::kernel_t<backend_traits, implementation_traits>::config(3),
    nArgs_(0),
    dimsGlobal_(global),
    dimsGroup_(group),
    memShared_(shared)
{
}

inline
unsigned
kernel_t::config::get_nargs() const
{
    return nArgs_;
}

inline
const dim3 &
kernel_t::config::get_dims_global() const
{
    return dimsGlobal_;
}

inline
const dim3 &
kernel_t::config::get_dims_group() const
{
    return dimsGroup_;
}

inline
gmacError_t
kernel_t::config::set_arg(const void *arg, size_t size, unsigned index)
{
    gmacError_t ret = gmacSuccess;

    params_[index] = arg;
    if (index >= nArgs_) {
        nArgs_ = index + 1;
    }

    return ret;
}

}}}

#endif /* TYPES_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
