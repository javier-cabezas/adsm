#ifndef GMAC_HAL_CUDA_KERNEL_IMPL_H_
#define GMAC_HAL_CUDA_KERNEL_IMPL_H_

namespace __impl { namespace hal { namespace cuda { namespace code {

inline
kernel::kernel(CUfunction func, const std::string &name) :
    parent(name),
    kernel_(func)
{
}

inline
CUfunction &
kernel::operator()()
{
    return kernel_;
}

inline
const CUfunction &
kernel::operator()() const
{
    return kernel_;
}

#if 0
inline 
kernel::launch &
kernel::launch_config(parent::config &conf, parent::arg_list &args, stream &stream)
{
    return *(new launch(*this, (kernel::config &) conf,
                               (kernel::arg_list &) args, stream));
}
#endif

inline
kernel::launch::launch(kernel &parent, config &conf, arg_list &args, stream &s) :
    parent::launch(parent),
    config_(conf),
    args_(args),
    stream_(s)
{
}

inline
const kernel::config &
kernel::launch::get_config() const
{
    return config_;
}

inline
const kernel::arg_list &
kernel::launch::get_arg_list() const
{
    return args_;
}

inline
const kernel &
kernel::launch::get_kernel() const
{
    return reinterpret_cast<const kernel &>(parent::launch::get_kernel());
}

inline
stream &
kernel::launch::get_stream()
{
    return stream_;
}

inline
const stream &
kernel::launch::get_stream() const
{
    return stream_;
}

inline
unsigned
kernel::arg_list::get_nargs() const
{
    return nArgs_;
}

inline
gmacError_t
kernel::arg_list::push_arg(const void *arg, size_t size)
{
    gmacError_t ret = gmacSuccess;

    params_[nArgs_++] = arg;

    return ret;
}

inline
hal_event_ptr
kernel::launch::execute(list_event_detail &_dependencies, gmacError_t &err)
{
    hal_event_ptr ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    get_stream().set_barrier(dependencies);

    if (err == gmacSuccess) {
        ret = execute(err);
    }

    return ret;
}

inline
hal_event_ptr
kernel::launch::execute(hal_event_ptr event, gmacError_t &err)
{
    list_event dependencies;
    dependencies.add_event(event);

    return execute(dependencies, err);
}

inline
hal_event_ptr
kernel::launch::execute(gmacError_t &err)
{
    get_stream().get_aspace().set();

    CUresult res;

    dim3 dimsGlobal = get_config().get_dims_global();
    dim3 dimsGroup = get_config().get_dims_group();

    TRACE(LOCAL, "kernel launch on stream: %p", get_stream()());
    event_ptr ret = create_event(true, _event_t::Kernel, get_stream().get_aspace());

    auto op = [&](CUstream s) -> CUresult
              {
                  return cuLaunchKernel(get_kernel()(), dimsGlobal.x,
                                                        dimsGlobal.y,
                                                        dimsGlobal.z,
                                                        dimsGroup.x,
                                                        dimsGroup.y,
                                                        dimsGroup.z,
                                                        get_config().memShared_,
                                                        s,
                                                        (void **) get_arg_list().params_,
                                                        NULL);
              };

    res = ret->add_operation(ret, get_stream(), cuda::operation::func_op(std::cref(op)), cuda::operation::Kernel, true);
    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    }

    return ret;
}

inline
kernel::config::config(dim3 global, dim3 group, size_t shared, cudaStream_t tokens) :
    hal_kernel::config(3),
    dimsGlobal_(global),
    dimsGroup_(group),
    memShared_(shared)
{
}

inline
const dim3 &
kernel::config::get_dims_global() const
{
    return dimsGlobal_;
}

inline
const dim3 &
kernel::config::get_dims_group() const
{
    return dimsGroup_;
}

}}}}

#endif /* TYPES_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
