#include "types.h"

namespace __impl { namespace hal { namespace cuda {

queue_event::queue_event() :
    gmac::util::mutex("queue_event")
{
}

_event_t *
queue_event::pop()
{
    _event_t *ret = NULL;

    lock();
    if (size() > 0) {
        ret = Parent::front();
        Parent::pop();
    }
    unlock();

    return ret;
}

void
queue_event::push(_event_t &event)
{
    lock();
    Parent::push(&event);
    unlock();
}

event_t 
context_t::copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "H (%p) -> D copy ("FMT_SIZE" bytes) on stream: %p", src, count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    CUresult res;

    set();

    ret.begin(stream);
    res = cuMemcpyHtoD(dst.get(), src, count);
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "D -> H (%p) copy ("FMT_SIZE" bytes) on stream: %p", dst, count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    CUresult res;

    set();

    ret.begin(stream);
    res = cuMemcpyDtoH(dst, src.get(), count);
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "D -> D copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    CUresult res;

    set();

    ret.begin(stream);
    res = cuMemcpyDtoD(dst.get(), src.get(), count);
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t
context_t::copy(accptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, input, count, stream, err);
    }

    return ret;
}

event_t
context_t::copy(accptr_t dst, device_input &input, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy(dst, input, count, stream, err);
    }

    return ret;
}

event_t
context_t::copy(accptr_t dst, device_input &input, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "IO -> D copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    hostptr_t host = get_memory(count);

    bool ok = input.read(host, count);

    if (ok) {
        CUresult res;

        set();

        ret.begin(stream);
        res = cuMemcpyHtoD(dst.get(), host, count);
        ret.end();

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

event_t
context_t::copy(device_output &output, accptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy(output, src, count, stream, err);
    }

    return ret;
}

event_t
context_t::copy(device_output &output, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy(output, src, count, stream, err);
    }

    return ret;
}

event_t
context_t::copy(device_output &output, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "D -> IO copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    hostptr_t host = get_memory(count);

    CUresult res;

    set();

    ret.begin(stream);
    res = cuMemcpyDtoH(host, src.get(), count);
    ret.end();

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

event_t 
context_t::copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, src, off, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, src, off, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "H (%p) -> D copy_async ("FMT_SIZE" bytes) on stream: %p", src.get_addr() + off, count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    CUresult res;

    set();

    ret.begin(stream);
    res = cuMemcpyHtoDAsync(dst.get(), src.get_addr() + off, count, stream());
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, off, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, off, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "D -> H (%p) copy_async ("FMT_SIZE" bytes) on stream: %p", dst.get_addr(), count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    CUresult res;

    set();

    ret.begin(stream);
    res = cuMemcpyDtoHAsync(dst.get_addr() + off, src.get(), count, stream());
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, src, count, stream, err);
    }

    return ret;
}

event_t 
context_t::copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "D -> D copy_async ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    CUresult res;

    set();

    ret.begin(stream);
    res = cuMemcpyDtoDAsync(dst.get(), src.get(), count, stream());
    ret.end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t
context_t::copy_async(accptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, input, count, stream, err);
    }

    return ret;
}

event_t
context_t::copy_async(accptr_t dst, device_input &input, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy_async(dst, input, count, stream, err);
    }

    return ret;
}

event_t
context_t::copy_async(accptr_t dst, device_input &input, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "IO -> D async copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    buffer_t &buffer = get_output_buffer(count);

    bool ok = input.read(buffer.get_addr(), count);

    if (ok) {
        CUresult res;

        set();

        ret.begin(stream);
        res = cuMemcpyHtoDAsync(dst.get(), buffer.get_addr(), count, stream());
        ret.end();

        err = error(res);
        if (err != gmacSuccess) {
            ret.reset();
        } else {
            stream.set_last_event(ret);
            ret.add_observer(buffer);
        }
    }

    return ret;
}

event_t
context_t::copy_async(device_output &output, accptr_t src, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = copy_async(output, src, count, stream, err);
    }

    return ret;
}

event_t
context_t::copy_async(device_output &output, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = copy_async(output, src, count, stream, err);
    }

    return ret;
}

event_t
context_t::copy_async(device_output &output, accptr_t src, size_t count, stream_t &stream, gmacError_t &err)
{
    TRACE(LOCAL, "D -> IO async copy ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    hostptr_t host = get_memory(count);

    CUresult res;

    set();

    ret.begin(stream);
    res = cuMemcpyDtoH(host, src.get(), count);
    ret.end();

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

event_t 
context_t::memset(accptr_t dst, int c, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = memset(dst, c, count, stream, err);
    }

    return ret;
}

event_t 
context_t::memset(accptr_t dst, int c, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = memset(dst, c, count, stream, err);
    }

    return ret;
}

event_t 
context_t::memset(accptr_t dst, int c, size_t count, stream_t &stream, gmacError_t &err)
{
    set();

    TRACE(LOCAL, "memset ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(false, _event_t::Transfer, *this);

    ret.begin(stream);
    CUresult res = cuMemsetD8(dst.get(), (unsigned char)c, count);
    ret.end();

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

event_t 
context_t::memset_async(accptr_t dst, int c, size_t count, stream_t &stream, list_event_detail &_dependencies, gmacError_t &err)
{
    event_t ret;
    list_event &dependencies = reinterpret_cast<list_event &>(_dependencies);
    err = dependencies.sync();

    if (err == gmacSuccess) {
        ret = memset_async(dst, c, count, stream, err);
    }

    return ret;
}

event_t 
context_t::memset_async(accptr_t dst, int c, size_t count, stream_t &stream, event_t event, gmacError_t &err)
{
    event_t ret;

    err = event.sync();

    if (err == gmacSuccess) {
        ret = memset_async(dst, c, count, stream, err);
    }

    return ret;
}

event_t 
context_t::memset_async(accptr_t dst, int c, size_t count, stream_t &stream, gmacError_t &err)
{
    set();

    TRACE(LOCAL, "memset_async ("FMT_SIZE" bytes) on stream: %p", count, stream());
    event_t ret(true, _event_t::Transfer, *this);

    ret.begin(stream);
    CUresult res = cuMemsetD8Async(dst.get(), (unsigned char)c, count, stream());
    ret.end();

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        stream.set_last_event(ret);
    }

    return ret;
}

hostptr_t
context_t::alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err)
{
    set();

    // TODO: add a parater to specify accesibility of the buffer from the device
    unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
    if (hint == GMAC_PROT_WRITE) {
        flags += CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    void *addr;
    CUresult res = cuMemHostAlloc(&addr, size, flags);
    err = cuda::error(res);

    //return new buffer_t(hostptr_t(addr), size, *this);
    return NULL;
}

buffer_t *
context_t::alloc_buffer(size_t size, GmacProtection hint, gmacError_t &err)
{
    set();

    // TODO: add a parater to specify accesibility of the buffer from the device
    unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
    if (hint == GMAC_PROT_WRITE) {
        flags += CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    void *addr;
    CUresult res = cuMemHostAlloc(&addr, size, flags);
    err = cuda::error(res);

    //return new buffer_t(hostptr_t(addr), size, *this);
    return NULL;
}

gmacError_t
context_t::free(accptr_t acc)
{
    set();

    CUresult ret = cuMemFree(acc.get());

    return cuda::error(ret);
}

gmacError_t
context_t::free_buffer(buffer_t &buffer)
{
    set();

    CUresult ret = cuMemFreeHost(buffer.get_addr());

    return cuda::error(ret);
}

}}}
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
