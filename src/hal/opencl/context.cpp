#include "types.h"

#include "device.h"

namespace __impl { namespace hal { namespace opencl {

_event_t *
context_t::get_new_event(bool async,_event_t::type t)
{
    _event_t *ret = queueEvents_.pop();
    if (ret == NULL) {
        ret = new _event_t(async, t, *this);
        TRACE(LOCAL, "Allocating new event %p\n", ret);
    } else {
        ret->reset(async, t);
        TRACE(LOCAL, "Reusing event %p\n", ret);
    }

    return ret;
}

event_t 
context_t::copy_backend(ptr_t dst, const ptr_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        nevents = dependencies->size();
        events  = dependencies->get_event_array();
    }

    event_t ret(false, _event_t::Transfer, *this);

    cl_int res;

    ret.begin(stream);
    if (dst.is_device_ptr() &&
        src.is_device_ptr()) {
        TRACE(LOCAL, "D -> D copy ("FMT_SIZE" bytes) on stream: %p", count, stream());

        if (dst.get_context()->get_device().has_direct_copy(src.get_context()->get_device())) {
            res = clEnqueueCopyBuffer(stream(), dst.get_device_addr(), src.get_device_addr(),
                                                dst.get_offset(),      src.get_offset(), count,
                                                nevents, events, &ret());
        } else {
            hostptr_t host = get_memory(count);

            res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_FALSE,
                                                src.get_offset(), count,
                                                host,
                                                nevents, events, &ret());
            if (res == CL_SUCCESS) {
                res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_TRUE,
                                                     dst.get_offset(), count,
                                                     host,
                                                     nevents, events, &ret());
            }

            put_memory(host, count);
        }
    } else if (dst.is_device_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> D copy ("FMT_SIZE" bytes) on stream: %p", src.get_host_addr(), count, stream());

        res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_TRUE,
                                             dst.get_offset(), count,
                                             src.get_host_addr(),
                                             nevents, events, &ret());
    } else if (dst.is_host_ptr() &&
               src.is_device_ptr()) {
        TRACE(LOCAL, "D -> H (%p) copy ("FMT_SIZE" bytes) on stream: %p", dst.get_host_addr(), count, stream());

        res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_TRUE,
                                            src.get_offset(), count,
                                            dst.get_host_addr(),
                                            nevents, events, &ret());
    } else if (dst.is_host_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy ("FMT_SIZE" bytes) on stream: %p", src.get_host_addr(), dst.get_host_addr(), count, stream());

        res = CL_SUCCESS;
        memcpy(dst.get_host_addr(), src.get_host_addr(), count);
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

event_t
context_t::copy_backend(ptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        nevents = dependencies->size();
        events  = dependencies->get_event_array();
    }

    TRACE(LOCAL, "IO -> D copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    hostptr_t host = get_memory(count);

    bool ok = input.read(host, count);

    if (ok) {
        cl_int res;

        ret.begin(stream);
        res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_TRUE,
                                             dst.get_offset(), count,
                                             host,
                                             nevents, events, &ret());

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

event_t
context_t::copy_backend(device_output &output, const ptr_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        nevents = dependencies->size();
        events  = dependencies->get_event_array();
    }

    TRACE(LOCAL, "D -> IO copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    hostptr_t host = get_memory(count);

    cl_int res;

    ret.begin(stream);

    res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_TRUE,
                                        src.get_offset(), count,
                                        host,
                                        nevents, events, &ret());


    bool ok = output.write(host, count);

    if (ok) {
        err = error(res);
    } else {
        err = gmacErrorIO;
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

event_t 
context_t::copy_async_backend(ptr_t dst, const ptr_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        nevents = dependencies->size();
        events  = dependencies->get_event_array();
    }

    event_t ret(true, _event_t::Transfer, *this);

    cl_int res;

    ret.begin(stream);
    if (dst.is_device_ptr() &&
        src.is_device_ptr()) {
        TRACE(LOCAL, "D (%p) -> D (%p) async copy ("FMT_SIZE" bytes) on stream: %p",
                     src.get_device_addr(),
                     dst.get_device_addr(),
                     count, stream());

        if (dst.get_context()->get_device().has_direct_copy(src.get_context()->get_device())) {
            res = clEnqueueCopyBuffer(stream(), dst.get_device_addr(), src.get_device_addr(),
                                                dst.get_offset(),      src.get_offset(), count,
                                                nevents, events, &ret());
        } else {
            hostptr_t mem = get_memory(count);

            res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_FALSE,
                                                src.get_offset(), count,
                                                mem,
                                                nevents, events, &ret());
            if (res == CL_SUCCESS) {
                res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_FALSE,
                                                     dst.get_offset(), count,
                                                     mem,
                                                     nevents, events, &ret());
            }

            if (res == CL_SUCCESS) {
                ret.add_trigger(do_member(context_t::put_memory, this, mem, count));
            }
        }
    } else if (dst.is_device_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> D (%p) async copy ("FMT_SIZE" bytes) on stream: %p",
                     src.get_host_addr(),
                     dst.get_device_addr(),
                     count, stream());
        //event_t last = stream.get_last_event();
        //if (last.is_valid()) {
        //    last.sync();
        //}

        //buffer_t &buffer = stream.get_buffer(count);

        //memcpy(buffer.get_addr(), src.get_host_addr(), count);

        res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_FALSE,
                                             dst.get_offset(), count,
                                             src.get_host_addr(),
                                             nevents, events, &ret());

        if (res == CL_SUCCESS) {
            // Release buffer after asynchronous copy
            //ret.add_trigger(do_member(stream_t::put_buffer, &stream, buffer));
        }
    } else if (dst.is_host_ptr() &&
               src.is_device_ptr()) {
        TRACE(LOCAL, "D (%p) -> H (%p) async copy ("FMT_SIZE" bytes) on stream: %p",
                     src.get_device_addr(),
                     dst.get_host_addr(),
                     count, stream());
        event_t last = stream.get_last_event();
        if (last.is_valid()) {
            last.sync();
        }

        //buffer_t &buffer = stream.get_buffer(count);

        res = clEnqueueReadBuffer(stream(), dst.get_device_addr(), CL_FALSE,
                                            dst.get_offset(), count,
                                            src.get_host_addr(),
                                            nevents, events, &ret());

        // Perform memcpy after asynchronous copy
        //ret.add_trigger(util::do_func(memcpy, buffer.get_addr(), src.get_host_addr(), count));
        // Release buffer after memcpy
        //ret.add_trigger(util::do_member(stream_t::put_buffer, &stream, buffer));
    } else if (dst.is_host_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy ("FMT_SIZE" bytes) on stream: %p",
                     src.get_host_addr(),
                     dst.get_host_addr(),
                     count, stream());

        res = CL_SUCCESS;
        memcpy(dst.get_host_addr(), src.get_host_addr(), count);
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

event_t
context_t::copy_async_backend(ptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        nevents = dependencies->size();
        events  = dependencies->get_event_array();
    }

    TRACE(LOCAL, "IO -> D async copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    //buffer_t &buffer = stream.get_buffer(count);
    hostptr_t mem = get_memory(count);

    bool ok = input.read(mem, count);

    if (ok) {
        cl_int res;

        ret.begin(stream);
        res = clEnqueueWriteBuffer(stream(), dst.get_device_addr(), CL_FALSE,
                                             dst.get_offset(), count,
                                             mem,
                                             nevents, events, &ret());

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

event_t
context_t::copy_async_backend(device_output &output, const ptr_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    cl_event *events = NULL;
    unsigned nevents = 0;

    if (_dependencies != NULL) {
        list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);
        nevents = dependencies->size();
        events  = dependencies->get_event_array();
    }

    TRACE(LOCAL, "D -> IO async copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    hostptr_t host = get_memory(count);

    cl_int res;

    ret.begin(stream);
    res = clEnqueueReadBuffer(stream(), src.get_device_addr(), CL_TRUE,
                                        src.get_offset(), count,
                                        host,
                                        nevents, events, &ret());

    bool ok = output.write(host, count);

    if (ok) {
        err = error(res);
    } else {
        err = gmacErrorIO;
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

event_t 
context_t::memset_backend(ptr_t dst, int c, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    FATAL("memset not implemented yet");
    return event_t();
#if 0
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        err = dependencies->sync();

        if (err != gmacSuccess) {
            return event_t();
        }
    }

    set();

    TRACE(LOCAL, "memset ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    ret.begin(stream);
    CUresult res = cuMemsetD8(dst.get_device_addr(), (unsigned char)c, count);
    ret.end();

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
#endif
}

event_t 
context_t::memset_async_backend(ptr_t dst, int c, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    FATAL("memset_async not implemented yet");
    return event_t();
#if 0
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        err = dependencies->sync();

        if (err != gmacSuccess) {
            return event_t();
        }
    }

    set();

    TRACE(LOCAL, "memset_async ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    ret.begin(stream);
    CUresult res = cuMemsetD8Async(dst.get_device_addr(), (unsigned char)c, count, stream());
    ret.end();

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
#endif
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
    hostptr_t hostPtr;
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
context_t::alloc_buffer(size_t size, GmacProtection hint, gmacError_t &err)
{
    FATAL("Not supported");

    return NULL;
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

}}}
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
