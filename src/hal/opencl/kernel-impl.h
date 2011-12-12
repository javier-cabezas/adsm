#ifndef GMAC_HAL_OPENCL_KERNEL_IMPL_H_
#define GMAC_HAL_OPENCL_KERNEL_IMPL_H_

namespace __impl { namespace hal { namespace opencl {

inline
kernel_t::kernel_t(cl_kernel func, const std::string &name) :
    Parent(func, name)
{
}

inline 
kernel_t::launch_ptr
kernel_t::launch_config(Parent::config &conf, Parent::arg_list &args, stream_t &stream)
{
    launch_ptr ret(new launch(*this, (kernel_t::config &) conf,
                              (kernel_t::arg_list &) args, stream));

    return ret;
}

inline
kernel_t::launch::launch(kernel_t &parent, config &conf, arg_list &args, stream_t &stream) :
    hal::detail::kernel_t<backend_traits, implementation_traits>::launch(parent, conf, args, stream)
{
}

inline
unsigned
kernel_t::arg_list::get_nargs() const
{
    return nArgs_;
}

inline
gmacError_t
kernel_t::arg_list::set_arg(kernel_t &k, const void *arg, size_t size, unsigned index)
{
    cl_int res;
    res = clSetKernelArg(k(), index, size, arg);
    nArgs_++;

    return error(res);
}

inline
event_t
kernel_t::launch::execute(list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);

    unsigned nelems;
    cl_event *evs = dependencies.get_event_array(get_stream(), nelems);

    ret = execute(nelems, evs, err);

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

    if (event.is_valid() == false) {
		ret = execute(err);
	} else {
		cl_event &ev = event();

    	ret = execute(1, &ev, err);

    	if (err == gmacSuccess) {
        	event.set_synced();
    	}
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

    const config &conf = get_config();

    const size_t *dimsGlobal = conf.get_dims_global();
    const size_t *dimsGroup  = conf.get_dims_group();

    TRACE(LOCAL, "kernel_launch<"FMT_ID">: launch on stream<"FMT_ID">",
                 get_print_id(), get_stream().get_print_id());
    event_t ret(true, _event_t::Kernel, get_stream().get_context());

    ret.begin(get_stream());
    cl_command_queue &stream = get_stream()();
    const cl_kernel &kernel = get_kernel()();
    cl_event  &ev     = ret();

    res = clEnqueueNDRangeKernel(stream,
                                 kernel,
                                 conf.get_ndims(),
                                 NULL,
                                 dimsGlobal,
                                 dimsGroup,
                                 nevents, events, &ev);

    err = error(res);

    if (err != gmacSuccess) {
        ret.invalidate();
    } else {
        get_stream().set_last_event(ret);
    }

    event_ = ret;

    return ret;
}

inline
kernel_t::config::config(unsigned ndims, const size_t *offset, const size_t *global, const size_t *group) :
    hal::detail::kernel_t<backend_traits, implementation_traits>::config(ndims),
    dimsOffset_(offset),
    dimsGlobal_(global),
    dimsGroup_(group)
{
}

inline
const size_t *
kernel_t::config::get_dims_offset() const
{
    return dimsOffset_;
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

}}}

#endif /* TYPES_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
