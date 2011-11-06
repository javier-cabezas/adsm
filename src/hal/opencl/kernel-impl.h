#ifndef GMAC_HAL_OPENCL_KERNEL_IMPL_H_
#define GMAC_HAL_OPENCL_KERNEL_IMPL_H_

namespace __impl { namespace hal { namespace opencl {

inline
kernel_t::kernel_t(cl_kernel func, const std::string &name) :
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
    hal::detail::kernel_t<device, backend_traits, implementation_traits>::launch(parent, dynamic_cast<config &>(conf), stream)
{
}

inline
event_t
kernel_t::launch::execute(list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);

    cl_event *evs = dependencies.get_event_array();

    ret = execute(unsigned(dependencies.size()), evs, err);

    if (err == gmacSuccess) {
        dependencies.set_synced();
    }

    if (evs != NULL) delete []evs;

    return ret;
}

inline
event_t
kernel_t::launch::execute(event_t event, gmacError_t &err)
{
    event_t ret;

    cl_event ev = event();

    ret = execute(1, &ev, err);

    if (err == gmacSuccess) {
        event.set_synced();
    }

    return ret;
}

inline
event_t
kernel_t::launch::execute(gmacError_t &err)
{
    return execute(0, NULL, err);
}

inline
event_t
kernel_t::launch::execute(unsigned nevents, const cl_event *events, gmacError_t &err)
{
    cl_int res;

    const size_t *dimsGlobal = get_config().get_dims_global();
    const size_t *dimsGroup = get_config().get_dims_group();

    TRACE(LOCAL, "kernel launch on stream: %p", get_stream()());
    event_t ret(true, _event_t::Kernel, get_stream().get_context());

    ret.begin(get_stream());
    res = clEnqueueNDRangeKernel(get_stream()(),
                                 get_kernel()(),
                                 get_config().get_ndims(),
                                 NULL,
                                 dimsGlobal,
                                 dimsGroup,
                                 nevents, events, &ret());

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    }

    return ret;
}



inline
kernel_t::config::config(unsigned ndims, const size_t *global, const size_t *group) :
    hal::detail::kernel_t<device, backend_traits, implementation_traits>::config(ndims),
    nArgs_(0),
    dimsGlobal_(global),
    dimsGroup_(group)
{
}

inline
unsigned
kernel_t::config::get_nargs() const
{
    return nArgs_;
}

inline
const size_t *
kernel_t::config::get_dims_global() const
{
    return dimsGlobal_;
}

inline
const size_t *
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
