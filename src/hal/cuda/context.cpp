#include "types.h"

#include "device.h"

namespace __impl { namespace hal { namespace cuda {

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

event_ptr 
context_t::copy_backend(ptr_t dst, ptr_const_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(stream);
    }

    event_ptr ret(false, get_event_type(dst, src), *this);

    CUresult res;

    set();

    ret->begin(stream);
    if (dst.is_device_ptr() &&
        src.is_device_ptr()) {
        TRACE(LOCAL, "D -> D copy ("FMT_SIZE" bytes) on stream: %p", count, stream());

        if (dst.get_context()->get_device().has_direct_copy(src.get_context()->get_device())) {
            res = cuMemcpyDtoD(dst.get_device_addr(), src.get_device_addr(), count);
        } else {
            host_ptr host = get_memory(count);

            res = cuMemcpyDtoH(host, src.get_device_addr(), count);
            if (res == CUDA_SUCCESS) {
                res = cuMemcpyHtoD(src.get_device_addr(), host, count);
            }

            put_memory(host, count);
        }
    } else if (dst.is_device_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> D copy ("FMT_SIZE" bytes) on stream: %p", src.get_host_addr(), count, stream());

        res = cuMemcpyHtoD(dst.get_device_addr(), src.get_host_addr(), count);
    } else if (dst.is_host_ptr() &&
               src.is_device_ptr()) {
        TRACE(LOCAL, "D -> H (%p) copy ("FMT_SIZE" bytes) on stream: %p", dst.get_host_addr(), count, stream());

        res = cuMemcpyDtoH(dst.get_host_addr(), src.get_device_addr(), count);
    } else if (dst.is_host_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy ("FMT_SIZE" bytes) on stream: %p", src.get_host_addr(), dst.get_host_addr(), count, stream());

        res = CUDA_SUCCESS;
        memcpy(dst.get_host_addr(), src.get_host_addr(), count);
    } else {
        FATAL("Unhandled case");
    }
    ret->end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_ptr
context_t::copy_backend(ptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(stream);
    }

    TRACE(LOCAL, "IO -> D copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_ptr ret(false, get_event_type(dst, input), *this);

    host_ptr host = get_memory(count);

    bool ok = input.read(host, count);

    if (ok) {
        CUresult res;

        set();

        ret->begin(stream);
        res = cuMemcpyHtoD(dst.get_device_addr(), host, count);
        ret->end();

        err = error(res);
        if (err != gmacSuccess) {
            ret.reset();
        } else {
            stream.set_last_event(ret);
        }
    }

    put_memory(host, count);

    return ret;
}

event_ptr
context_t::copy_backend(device_output &output, ptr_const_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(stream);
    }

    TRACE(LOCAL, "D -> IO copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_ptr ret(false, get_event_type(output, src), *this);

    host_ptr host = get_memory(count);

    CUresult res;

    set();

    ret->begin(stream);
    res = cuMemcpyDtoH(host, src.get_device_addr(), count);
    ret->end();

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

    return ret;
}

event_ptr 
context_t::copy_async_backend(ptr_t dst, ptr_const_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(stream);
    }

    event_ptr ret(true, get_event_type(dst, src), *this);

    CUresult res;

    set();

    ret->begin(stream);
    if (dst.is_device_ptr() &&
        src.is_device_ptr()) {
        TRACE(LOCAL, "D (%p) -> D (%p) async copy ("FMT_SIZE" bytes) on stream: "FMT_ID,
                     src.get_device_addr(),
                     dst.get_device_addr(),
                     count, stream.get_print_id());

        if (dst.get_context()->get_device().has_direct_copy(src.get_context()->get_device())) {
            res = cuMemcpyDtoDAsync(dst.get_device_addr(), src.get_device_addr(), count, stream());
        } else {
            buffer_t *buffer = get_input_buffer(count, stream, ret);

            res = cuMemcpyDtoHAsync(buffer->get_addr(), src.get_device_addr(), count, stream());
            if (res == CUDA_SUCCESS) {
                // TODO, check if an explicit synchronization is required
                res = cuMemcpyHtoDAsync(src.get_device_addr(), buffer->get_addr(), count, stream());
            }
        }
    } else if (dst.is_device_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> D (%p) async copy ("FMT_SIZE" bytes) on stream: "FMT_ID,
                     src.get_host_addr(),
                     dst.get_device_addr(),
                     count, stream.get_print_id());
        buffer_t *buffer = get_output_buffer(count, stream, ret);

        memcpy(buffer->get_addr(), src.get_host_addr(), count);

        res = cuMemcpyHtoDAsync(dst.get_device_addr(), src.get_host_addr(), count, stream());
    } else if (dst.is_host_ptr() &&
               src.is_device_ptr()) {
        TRACE(LOCAL, "D (%p) -> H (%p) async copy ("FMT_SIZE" bytes) on stream: "FMT_ID,
                     src.get_device_addr(),
                     dst.get_host_addr(),
                     count, stream.get_print_id());
        event_ptr last = stream.get_last_event();
        last->sync();

        buffer_t *buffer = get_input_buffer(count, stream, ret);

        res = cuMemcpyDtoHAsync(dst.get_host_addr(), src.get_device_addr(), count, stream());

        // Perform memcpy after asynchronous copy
        ret.add_trigger(do_func(memcpy, buffer->get_addr(), src.get_host_addr(), count));
    } else if (dst.is_host_ptr() &&
               src.is_host_ptr()) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy ("FMT_SIZE" bytes) on stream: "FMT_ID,
                     src.get_host_addr(),
                     dst.get_host_addr(),
                     count, stream.get_print_id());

        res = CUDA_SUCCESS;
        memcpy(dst.get_host_addr(), src.get_host_addr(), count);
    }

    ret->end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_ptr
context_t::copy_async_backend(ptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(stream);
    }

    TRACE(LOCAL, "IO -> D async copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_ptr ret(true, get_event_type(dst, input), *this);

    //buffer_t &buffer = stream.get_buffer(count);
    host_ptr mem = get_memory(count);

    bool ok = input.read(mem, count);

    if (ok) {
        CUresult res;

        set();

        ret->begin(stream);
        res = cuMemcpyHtoDAsync(dst.get_device_addr(), mem, count, stream());
        ret->end();

        err = error(res);
        if (err != gmacSuccess) {
            ret.reset();
        } else {
            stream.set_last_event(ret);
            //ret.add_trigger(do_member(stream_t::put_buffer, &stream, buffer));
        }
    }

    return ret;
}

event_ptr
context_t::copy_async_backend(device_output &output, ptr_const_t src, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(stream);
    }

    TRACE(LOCAL, "D -> IO async copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_ptr ret(true, get_event_type(output, src), *this);

    host_ptr host = get_memory(count);

    CUresult res;

    set();

    ret->begin(stream);
    res = cuMemcpyDtoHAsync(host, src.get_device_addr(), count, stream());
    ret->end();

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

    return ret;
}

event_ptr 
context_t::memset_backend(ptr_t dst, int c, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(stream);
    }

    set();

    TRACE(LOCAL, "memset ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_ptr ret(false, event_ptr::type::TransferToDevice, *this);

    ret->begin(stream);
    CUresult res = cuMemsetD8(dst.get_device_addr(), (unsigned char)c, count);
    ret->end();

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_ptr 
context_t::memset_async_backend(ptr_t dst, int c, size_t count, stream_t &stream, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(stream);
    }

    set();

    TRACE(LOCAL, "memset_async ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_ptr ret(true, _event_t::type::TransferToDevice, *this);

    ret->begin(stream);
    CUresult res = cuMemsetD8Async(dst.get_device_addr(), (unsigned char)c, count, stream());
    ret->end();

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

ptr_t
context_t::alloc(size_t count, gmacError_t &err)
{
    set();

    CUdeviceptr devPtr = 0;
    CUresult res = cuMemAlloc(&devPtr, count);

    err = cuda::error(res);

    return ptr_t(ptr_t::backend_ptr(devPtr), this);
}

ptr_t
context_t::alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err)
{
    set();

    // TODO: add a parater to specify accesibility of the buffer from the device
    unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
    if (hint == GMAC_PROT_WRITE) {
        flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    void *addr;
    CUresult res = cuMemHostAlloc(&addr, size, flags);
    err = cuda::error(res);

    return ptr_t(host_ptr(addr));
}

buffer_t *
context_t::alloc_buffer(size_t size, GmacProtection hint, stream_t &/*stream*/, gmacError_t &err)
{
    set();

    // TODO: add a parater to specify accesibility of the buffer from the device
    unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
    if (hint == GMAC_PROT_WRITE) {
        flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    void *addr;
    CUresult res = cuMemHostAlloc(&addr, size, flags);
    err = cuda::error(res);

    TRACE(LOCAL, "Created buffer: %p ("FMT_SIZE")", addr, size);
    buffer_t *ret = NULL;
    if (res == CUDA_SUCCESS) {
        ret = new buffer_t(host_ptr(addr), size, *this);
    }

    return ret;
}

gmacError_t
context_t::free(ptr_t acc)
{
    set();

    CUresult ret = cuMemFree(acc.get_device_addr());

    return cuda::error(ret);
}

gmacError_t
context_t::free_buffer(buffer_t &buffer)
{
    set();

    CUresult ret = cuMemFreeHost(buffer.get_addr());

    return cuda::error(ret);
}

gmacError_t
context_t::free_host_pinned(ptr_t ptr)
{
    set();

    CUresult ret = cuMemFreeHost(ptr.get_host_addr());

    return cuda::error(ret);
}

}}}
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
