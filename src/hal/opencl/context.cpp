#include "kernels/kernels.h"

#include "types.h"

#include "device.h"

namespace __impl { namespace hal { namespace opencl {

static event_ptr::type
get_event_type(ptr_t dst, ptr_const_t src)
{
    if (dst.is_device_ptr() &&
        src.is_device_ptr()) {
        return event_ptr::type::TransferDevice;
    } else if (dst.is_device_ptr() &&
               src.is_host_ptr()) {
        return event_ptr::type::TransferToDevice;
    } else if (dst.is_host_ptr() &&
               src.is_device_ptr()) {
        return event_ptr::type::TransferToHost;
    } else {
        return event_ptr::type::TransferHost;
    }
}

static event_ptr::type
get_event_type(ptr_t dst, device_input &/* input */)
{
    if (dst.is_device_ptr()) {
        return event_ptr::type::TransferToDevice;
    } else {
        return event_ptr::type::TransferToHost;
    }
}

static event_ptr::type
get_event_type(device_output &/* output */, ptr_const_t src)
{
    return event_ptr::type::TransferToHost;
}

context_t::context_t(cl_context ctx, device &dev) :
    Parent(ctx, dev)
{
    TRACE(LOCAL, "context<"FMT_ID">: creating context: ", get_print_id());
}

_event_t *
context_t::get_new_event(bool async,_event_t::type t)
{
    _event_t *ret = queueEvents_.pop();
    if (ret == NULL) {
        ret = new _event_t(async, t, *this);
        TRACE(LOCAL, "context<"FMT_ID">: allocating new event "FMT_ID,
                     get_print_id(), ret->get_print_id());
    } else {
        ret->reset(async, t);
        TRACE(LOCAL, "context<"FMT_ID">: reusing event "FMT_ID,
                     get_print_id(), ret->get_print_id());
    }

    return ret;
}

void
context_t::dispose_event(_event_t &event)
{
    TRACE(LOCAL, "context<"FMT_ID">: disposing event "FMT_ID,
                 get_print_id(), event.get_print_id());
    queueEvents_.push(event);
}



event_ptr 
context_t::copy_backend(ptr_t dst, ptr_const_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        events = dependencies->get_event_array(stream, nevents);
    }

    event_ptr ret(false, get_event_type(dst, src), *this);

    cl_int res;

    ret->begin(stream);
    if (dst.is_device_ptr() &&
        src.is_device_ptr()) {
        TRACE(LOCAL, "D -> D copy ("FMT_SIZE" bytes) on stream: "FMT_ID, count, stream.get_print_id());

        if (dst.get_context()->get_device().has_direct_copy(src.get_context()->get_device())) {
            res = clEnqueueCopyBuffer(stream(), dst.get_device_addr(), src.get_device_addr(),
                                                dst.get_offset(),      src.get_offset(), count,
                                                nevents, events, &(*ret)());
        } else {
            host_ptr host = get_memory(count);

            res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_FALSE,
                                                src.get_offset(), count,
                                                host,
                                                nevents, events, &(*ret)());
            if (res == CL_SUCCESS) {
                res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_TRUE,
                                                     dst.get_offset(), count,
                                                     host,
                                                     nevents, events, &(*ret)());
            }

            put_memory(host, count);
        }
    } else if (dst.is_device_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> D copy ("FMT_SIZE" bytes) on stream: "FMT_ID, src.get_host_addr(), count, stream.get_print_id());

        res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_TRUE,
                                             dst.get_offset(), count,
                                             src.get_host_addr(),
                                             nevents, events, &(*ret)());
    } else if (dst.is_host_ptr() &&
               src.is_device_ptr()) {
        TRACE(LOCAL, "D -> H (%p) copy ("FMT_SIZE" bytes) on stream: "FMT_ID, dst.get_host_addr(), count, stream.get_print_id());

        res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_TRUE,
                                            src.get_offset(), count,
                                            dst.get_host_addr(),
                                            nevents, events, &(*ret)());
    } else if (dst.is_host_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy ("FMT_SIZE" bytes) on stream: "FMT_ID, src.get_host_addr(), dst.get_host_addr(), count, stream.get_print_id());

        res = CL_SUCCESS;
        ::memcpy(dst.get_host_addr(), src.get_host_addr(), count);
    } else {
        FATAL("Unhandled case");
    }

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    if (nevents > 0) {
        delete []events;
    }

    return ret;
}

event_ptr
context_t::copy_backend(ptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        events = dependencies->get_event_array(stream, nevents);
    }

    TRACE(LOCAL, "IO -> D copy ("FMT_SIZE" bytes) on stream: "FMT_ID, count, stream.get_print_id());
    event_ptr ret(false, get_event_type(dst, input), *this);

    host_ptr host = get_memory(count);

    bool ok = input.read(host, count);

    if (ok) {
        cl_int res;

        ret->begin(stream);
        res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_TRUE,
                                             dst.get_offset(), count,
                                             host,
                                             nevents, events, &(*ret)());

        err = error(res);
        if (err != gmacSuccess) {
            ret.reset();
        } else {
            stream.set_last_event(ret);
        }

        if (nevents > 0) {
            delete []events;
        }
    }

    put_memory(host, count);

    return ret;
}

event_ptr
context_t::copy_backend(device_output &output, ptr_const_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        events = dependencies->get_event_array(stream, nevents);
    }

    TRACE(LOCAL, "D -> IO copy ("FMT_SIZE" bytes) on stream: "FMT_ID, count, stream.get_print_id());
    event_ptr ret(false, get_event_type(output, src), *this);

    host_ptr host = get_memory(count);

    cl_int res;

    ret->begin(stream);

    res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_TRUE,
                                        src.get_offset(), count,
                                        host,
                                        nevents, events, &(*ret)());


    if (err == gmacSuccess) {
        bool ok = output.write(host, count);

        if (ok) {
            err = error(res);
        } else {
            err = gmacErrorIO;
        }
    }

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    put_memory(host, count);

    if (nevents > 0) {
        delete []events;
    }

    return ret;
}

event_ptr 
context_t::copy_async_backend(ptr_t dst, ptr_const_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        events = dependencies->get_event_array(stream, nevents);
    }

    event_ptr ret(true, get_event_type(dst, src), *this);

    cl_int res;

    ret->begin(stream);
    if (dst.is_device_ptr() &&
        src.is_device_ptr()) {
        TRACE(LOCAL, "D (%p) -> D (%p) async copy ("FMT_SIZE" bytes) on stream: "FMT_ID,
                     src.get_device_addr(),
                     dst.get_device_addr(),
                     count, stream.get_print_id());

        if (dst.get_context()->get_device().has_direct_copy(src.get_context()->get_device())) {
            res = clEnqueueCopyBuffer(stream(), dst.get_device_addr(), src.get_device_addr(),
                                                dst.get_offset(),      src.get_offset(), count,
                                                nevents, events, &(*ret)());
        } else {
            buffer_t *buffer = get_input_buffer(count, stream, ret);

            res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_FALSE,
                                                src.get_offset(), count,
                                                buffer->get_addr(),
                                                nevents, events, &(*ret)());
            if (res == CL_SUCCESS) {
                res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_FALSE,
                                                     dst.get_offset(), count,
                                                     buffer->get_addr(),
                                                     nevents, events, &(*ret)());
            }
        }
    } else if (dst.is_device_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> D (%p) async copy ("FMT_SIZE" bytes) on stream: "FMT_ID,
                     src.get_host_addr(),
                     dst.get_device_addr(),
                     count, stream.get_print_id());
        buffer_t *buffer = get_output_buffer(count, stream, ret);

        ::memcpy(buffer->get_addr(), src.get_host_addr(), count);

        res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_FALSE,
                                             dst.get_offset(), count,
                                             buffer->get_addr(),
                                             nevents, events, &(*ret)());

        if (res == CL_SUCCESS) {
            // Release buffer after asynchronous copy
            //ret.add_trigger(do_member(stream_t::put_buffer, &stream, buffer));
        }
    } else if (dst.is_host_ptr() &&
               src.is_device_ptr()) {
        TRACE(LOCAL, "D (%p) -> H (%p) async copy ("FMT_SIZE" bytes) on stream: "FMT_ID,
                     src.get_device_addr(),
                     dst.get_host_addr(),
                     count, stream.get_print_id());
        buffer_t *buffer = get_input_buffer(count, stream, ret);

        res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_FALSE,
                                            src.get_offset(), count,
                                            buffer->get_addr(),
                                            nevents, events, &(*ret)());

        // Perform memcpy after asynchronous copy
        ret.add_trigger(do_func(::memcpy, dst.get_host_addr(), buffer->get_addr(), count));
        // Release buffer after memcpy
        //ret.add_trigger(util::do_member(stream_t::put_buffer, &stream, buffer));
    } else if (dst.is_host_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy ("FMT_SIZE" bytes) on stream: "FMT_ID,
                     src.get_host_addr(),
                     dst.get_host_addr(),
                     count, stream.get_print_id());

        res = CL_SUCCESS;
        ::memcpy(dst.get_host_addr(), src.get_host_addr(), count);
    }

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    if (nevents > 0) {
        delete []events;
    }

    return ret;
}

event_ptr
context_t::copy_async_backend(ptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        events = dependencies->get_event_array(stream, nevents);
    }

    TRACE(LOCAL, "IO -> D async copy ("FMT_SIZE" bytes) on stream: "FMT_ID, count, stream.get_print_id());
    event_ptr ret(true, get_event_type(dst, input), *this);

    //buffer_t &buffer = stream.get_buffer(count);
    host_ptr mem = get_memory(count);

    bool ok = input.read(mem, count);

    if (ok) {
        cl_int res;

        ret->begin(stream);
        res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_FALSE,
                                             dst.get_offset(), count,
                                             mem,
                                             nevents, events, &(*ret)());

        err = error(res);
        if (err != gmacSuccess) {
            ret.reset();
        } else {
            stream.set_last_event(ret);
            ret.add_trigger(do_member(context_t::put_memory, this, mem, count));
            //ret.add_trigger(do_member(stream_t::put_buffer, &stream, buffer));
        }

        if (nevents > 0) {
            delete []events;
        }
    }

    return ret;
}

event_ptr
context_t::copy_async_backend(device_output &output, ptr_const_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        events = dependencies->get_event_array(stream, nevents);
    }

    TRACE(LOCAL, "D -> IO async copy ("FMT_SIZE" bytes) on stream: "FMT_ID, count, stream.get_print_id());
    event_ptr ret(false, get_event_type(output, src), *this);

    host_ptr host = get_memory(count);

    cl_int res;

    ret->begin(stream);
    res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_TRUE,
                                        src.get_offset(), count,
                                        host,
                                        nevents, events, &(*ret)());

    if (err == gmacSuccess) {
        bool ok = output.write(host, count);

        if (ok) {
            err = error(res);
        } else {
            err = gmacErrorIO;
        }
    }

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    put_memory(host, count);

    if (nevents > 0) {
        delete []events;
    }

    return ret;
}

event_ptr 
context_t::memset_backend(ptr_t dst, int c, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    FATAL("Not implemented");
#if 0
    kernel_t *kernel = get_code_repository().get_kernel(gmac_memset);
    ASSERTION(kernel, "memset kernel not found");

    kernel_t::arg_list args;
    cl_mem devptr = dst.get_device_addr();
    unsigned offset = unsigned(dst.get_offset());
    unsigned ucount = unsigned(count);

    gmacError_t err2;
    err2 = args.set_arg(*kernel, &devptr, sizeof(cl_mem), 0);
    ASSERTION(err2 == gmacSuccess, "error setting parameter for memset");
    err2 = args.set_arg(*kernel, &offset, sizeof(unsigned), 1);
    ASSERTION(err2 == gmacSuccess, "error setting parameter for memset");
    err2 = args.set_arg(*kernel, &ucount, sizeof(unsigned), 2);
    ASSERTION(err2 == gmacSuccess, "error setting parameter for memset");

    size_t global = count / 4;
    kernel_t::config config(1, NULL, &global, NULL);

    kernel_t::launch_ptr launch = kernel->launch_config(config, args, stream);
#endif

    event_ptr ret;
#if 0
    if (_dependencies) {
        ret = launch->execute(*_dependencies, err2);
    } else {
        ret = launch->execute(err2);
    }

    ASSERTION(err2 == gmacSuccess, "error launching memset");
    err2 = ret.sync();
    ASSERTION(err2 == gmacSuccess, "error launching memset");
#endif

    stream.set_last_event(ret);

    err = gmacSuccess;

    return ret;
}

event_ptr 
context_t::memset_async_backend(ptr_t dst, int c, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    kernel_t *kernel = get_code_repository().get_kernel(gmac_memset);
    ASSERTION(kernel, "memset kernel not found");

    kernel_t::arg_list args;
    cl_mem devptr = dst.get_device_addr();
    unsigned offset = unsigned(dst.get_offset());
    unsigned ucount = unsigned(count);

    gmacError_t err2;
    err2 = args.set_arg(*kernel, &devptr, sizeof(cl_mem), 0);
    ASSERTION(err2 == gmacSuccess, "error setting parameter for memset");
    err2 = args.set_arg(*kernel, &offset, sizeof(unsigned), 1);
    ASSERTION(err2 == gmacSuccess, "error setting parameter for memset");
    err2 = args.set_arg(*kernel, &ucount, sizeof(unsigned), 2);
    ASSERTION(err2 == gmacSuccess, "error setting parameter for memset");

    size_t global = count / 4;
    kernel_t::config config(1, NULL, &global, NULL);

    kernel_t::launch_ptr launch = kernel->launch_config(config, args, stream);

    event_ptr ret;
    if (_dependencies) {
        ret = launch->execute(*_dependencies, err2);
    } else {
        ret = launch->execute(err2);
    }

    ASSERTION(err2 == gmacSuccess, "error executing memset");
    err2 = ret->sync();
    ASSERTION(err2 == gmacSuccess, "error executing memset");

    err = gmacSuccess;

    return ret;
}

ptr_t
context_t::alloc(size_t size, gmacError_t &err)
{
    cl_mem devPtr;
    cl_int res;

    devPtr = clCreateBuffer((*this)(), CL_MEM_READ_WRITE, size, NULL, &res);

    err = error(res);

    return ptr_t(devPtr, this);
}


ptr_t
context_t::alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err)
{
    FATAL("Not supported");

    return ptr_t();
#if 0
    cl_mem devPtr;
    host_ptr hostPtr;
    cl_int res;

    unsigned flags = CL_MEM_ALLOC_HOST_PTR;
    if (hint == GMAC_PROT_WRITE) {
        flags |= CL_MEM_READ_ONLY;
    } else {
        flags |= CL_MEM_READ_WRITE;
    }
    devPtr = clCreateBuffer((*this)(), flags, size, NULL, &res);

    if (res == CL_SUCCESS) {
            res = clEnqueueMapBuffer(
                                     devPtr, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                     );
    }

    err = error(res);

    return ptr_t(devPtr, this);
#endif
}

buffer_t *
context_t::alloc_buffer(size_t size, GmacProtection hint, stream_t &stream, gmacError_t &err)
{
    cl_int res;
    host_ptr hostPtr = 0;

    // TODO: add a parater to specify accesibility of the buffer from the device
    cl_mem_flags flags = CL_MEM_ALLOC_HOST_PTR;
    if (hint == GMAC_PROT_WRITE) {
        flags |= CL_MEM_READ_ONLY;
        TRACE(LOCAL, "alloc out buffer on context "FMT_ID, this->get_print_id());
    } else if (hint == GMAC_PROT_READ) {
        flags |= CL_MEM_WRITE_ONLY;
        TRACE(LOCAL, "alloc in  buffer on context "FMT_ID, this->get_print_id());
    } else {
        flags |= CL_MEM_READ_WRITE;
        TRACE(LOCAL, "alloc inout buffer on context "FMT_ID, this->get_print_id());
    }

    cl_mem devPtr = clCreateBuffer((*this)(), flags, size, NULL, &res);

    if (res == CL_SUCCESS) {
        cl_map_flags mapFlags = 0;

        if (hint == GMAC_PROT_WRITE) {
            mapFlags |= CL_MAP_WRITE;
        } else if (hint == GMAC_PROT_READ) {
            mapFlags |= CL_MAP_READ;
        } else {
            mapFlags |= CL_MAP_READ | CL_MAP_WRITE;
        }

        TRACE(LOCAL, "mapping buffer on context "FMT_ID" using stream "FMT_ID, this->get_print_id(), stream.get_print_id());

        hostPtr = (host_ptr) clEnqueueMapBuffer(stream(), devPtr, CL_TRUE, mapFlags, 0, size, 0, NULL, NULL, &res);
    }

    buffer_t *ret = NULL;
    if (res == CL_SUCCESS) {
        ret = new buffer_t(host_ptr(hostPtr), devPtr, size, *this);
    }

    ASSERTION(res == CL_SUCCESS);

    err = error(res);

    return ret;
}

gmacError_t
context_t::free(ptr_t ptr)
{
    cl_int res = clReleaseMemObject(ptr.get_device_addr());

    return error(res);
}

gmacError_t
context_t::free_buffer(buffer_t &buffer)
{
    FATAL("Not supported yet");

    return gmacErrorFeatureNotSupported;
}

gmacError_t
context_t::free_host_pinned(ptr_t ptr)
{
    FATAL("Not supported yet");

    return gmacErrorFeatureNotSupported;
}

code_repository &
context_t::get_code_repository()
{
    code_repository *repository;
    platform &plat = get_device().get_platform();
    map_platform_repository::iterator it = Modules_.find(&plat);
    if (it == Modules_.end()) {
        gmacError_t err;
        code_repository tmp = module_descriptor::create_modules(plat, err);
        ASSERTION(err == gmacSuccess);
        repository = &Modules_.insert(map_platform_repository::value_type(&plat, tmp)).first->second;
    } else {
        repository = &it->second;
    }

    return *repository;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
