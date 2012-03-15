#include "util/misc.h"

#include "hal/cuda/types.h"

#include "hal/cuda/phys/processing_unit.h"

namespace __impl { namespace hal { namespace cuda { namespace virt {

template <typename Ptr>
static bool
is_host_ptr(Ptr &&p)
{
    detail::virt::object &o = p.get_view().get_object();

    const detail::phys::memory &m = o.get_memory();

    return util::algo::has_predicate(m.get_attached_units(),
                                     [](detail::phys::processing_unit *pu) -> bool
                                     {
                                         return pu->get_type() == detail::phys::processing_unit::PUNIT_TYPE_CPU;
                                     });
}

template <typename Ptr>
static bool
is_device_ptr(Ptr &&p)
{
    detail::virt::object &o = p.get_view().get_object();

    const detail::phys::memory &m = o.get_memory();

    return util::algo::has_predicate(m.get_attached_units(),
                                     [](const detail::phys::processing_unit *pu) -> bool
                                     {
                                         return pu->get_type() == phys::processing_unit::PUNIT_TYPE_GPU;
                                     });
}

template <typename Ptr>
static CUdeviceptr
get_cuda_ptr(Ptr &&p)
{
    return CUdeviceptr(p.get_view().get_offset() + p.get_offset());
}

static void *
get_host_ptr(hal::ptr p)
{
    return (void *)(p.get_view().get_offset() + p.get_offset());
}

static const void *
get_host_ptr(hal::const_ptr p)
{
    return (const void *)(p.get_view().get_offset() + p.get_offset());
}

static hal_event::type
get_event_type(hal::ptr dst, hal::const_ptr src)
{

    if (is_device_ptr(dst) &&
        is_device_ptr(src)) {
        return hal_event::type::TransferDevice;
    } else if (is_device_ptr(dst) &&
               is_host_ptr(src)) {
        return hal_event::type::TransferToDevice;
    } else if (is_host_ptr(dst) &&
               is_device_ptr(src)) {
        return hal_event::type::TransferToHost;
    } else {
        return hal_event::type::TransferHost;
    }
}

static hal_event::type
get_event_type(hal::ptr dst, device_input &/* input */)
{
    if (is_device_ptr(dst)) {
        return hal_event::type::TransferToDevice;
    } else {
        return hal_event::type::TransferToHost;
    }
}

static hal_event::type
get_event_type(device_output &/* output */, hal::const_ptr src)
{
    return hal_event::type::TransferToHost;
}

template <typename Ptr1, typename Ptr2>
static aspace &
get_default_aspace(Ptr1 &dst, Ptr2 &src)
{
    if (is_device_ptr(dst) &&
        is_device_ptr(src)) {
        return *dst.get_aspace();
    } else if (is_device_ptr(dst) &&
               is_host_ptr(src)) {
        return *dst.get_aspace();
    } else if (is_host_ptr(dst) &&
               is_device_ptr(src)) {
        return *src.get_aspace();
    } else {
        return *dst.get_aspace();
    }
}

_event_t *
aspace::get_new_event(bool async,_event_t::type t)
{
    _event_t *ret = reinterpret_cast<_event_t *>(queueEvents_.pop());
    if (ret == NULL) {
        ret = new _event_t(async, t, *this);
    } else {
        ret->reset(async, t);
    }

    return ret;
}

void
aspace::dispose_event(_event_t &event)
{
    queueEvents_.push(event);
}

aspace::aspace(phys::hal_processing_unit &_pu, phys::aspace &pas, gmacError_t &err) :
    parent(_pu, pas, err)
{
    TRACE(LOCAL, FMT_ID2" Created", get_print_id2());

    phys::processing_unit &pu = reinterpret_cast<phys::processing_unit &>(_pu);

    CUcontext ctx, tmp;
    unsigned int flags = 0;
#if CUDA_VERSION >= 2020
    if (pu.get_major() >= 2 || (pu.get_major() == 1 && pu.get_minor() >= 1)) {
        flags |= CU_CTX_MAP_HOST;
    }
#else
    TRACE(LOCAL,"Host mapped memory not supported by the HW");
#endif
    CUresult res = cuCtxCreate(&ctx, flags, pu.get_cuda_id());
    if (res != CUDA_SUCCESS) {
        err = error(res);
        return;
    }
    res = cuCtxPopCurrent(&tmp);
    if (res != CUDA_SUCCESS) {
        err = error(res);
        return;
    }
    context_ = ctx;
}

static
cuda::map_context_repository Modules_("map_context_repository");

hal_code_repository &
aspace::get_code_repository()
{
    code_repository *repository;
    map_context_repository::iterator it = Modules_.find(this);
    if (it == Modules_.end()) {
        set();

        repository = module_descriptor::create_modules();
        Modules_.insert(map_context_repository::value_type(this, repository));
    } else {
        repository = it->second;
    }

    return *repository;
}

hal_event_ptr 
aspace::copy(hal::ptr dst, hal::const_ptr src, size_t count, hal_stream &_s, list_event_detail *_dependencies, gmacError_t &err)
{
    stream &s = reinterpret_cast<stream &>(_s);
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(s);
    }

    set();

    event_ptr ret = create_event(false, get_event_type(dst, src), *this);

    CUresult res;

    ret->begin(s);
    if (is_device_ptr(dst) &&
        is_device_ptr(src)) {
        TRACE(LOCAL, "D -> D copy ("FMT_SIZE" bytes) on stream: %p", count, s());

        if (has_direct_copy(src, dst)) {
            res = cuMemcpyDtoD(get_cuda_ptr(dst),
                               get_cuda_ptr(src), count);
        } else {
            host_ptr host = get_memory(count);

            res = cuMemcpyDtoH(host, get_cuda_ptr(src), count);
            if (res == CUDA_SUCCESS) {
                res = cuMemcpyHtoD(get_cuda_ptr(dst), host, count);
            }

            put_memory(host, count);
        }
    } else if (is_device_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> D copy ("FMT_SIZE" bytes) on stream: %p", get_host_ptr(src), count, s());

        res = cuMemcpyHtoD(get_cuda_ptr(dst), get_host_ptr(src), count);
    } else if (is_host_ptr(dst) &&
               is_device_ptr(src)) {
        TRACE(LOCAL, "D -> H (%p) copy ("FMT_SIZE" bytes) on stream: %p", get_host_ptr(dst), count, s());

        res = cuMemcpyDtoH(get_host_ptr(dst), get_cuda_ptr(src), count);
    } else if (is_host_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy ("FMT_SIZE" bytes) on stream: %p", get_host_ptr(src), get_host_ptr(dst), count, s());

        res = CUDA_SUCCESS;
        memcpy(get_host_ptr(dst), get_host_ptr(src), count);
    } else {
        FATAL("Unhandled case");
    }
    ret->end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        s.set_last_event(ret);
    }

    return ret;
}

hal_event_ptr
aspace::copy(hal::ptr dst, device_input &input, size_t count, hal_stream &_s, list_event_detail *_dependencies, gmacError_t &err)
{
    stream &s = reinterpret_cast<stream &>(_s);
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(s);
    }

    TRACE(LOCAL, "IO -> D copy ("FMT_SIZE" bytes) on stream: %p", count, s());
    event_ptr ret = create_event(false, get_event_type(dst, input), *this);

    host_ptr host = get_memory(count);

    bool ok = input.read(host, count);

    if (ok) {
        CUresult res;

        set();

        ret->begin(s);
        res = cuMemcpyHtoD(get_cuda_ptr(dst), host, count);
        ret->end();

        err = error(res);
        if (err != gmacSuccess) {
            ret.reset();
        } else {
            s.set_last_event(ret);
        }
    }

    put_memory(host, count);

    return ret;
}

hal_event_ptr
aspace::copy(device_output &output, hal::const_ptr src, size_t count, hal_stream &_s, list_event_detail *_dependencies, gmacError_t &err)
{
    stream &s = reinterpret_cast<stream &>(_s);
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(s);
    }

    TRACE(LOCAL, "D -> IO copy ("FMT_SIZE" bytes) on stream: %p", count, s());
    event_ptr ret = create_event(false, get_event_type(output, src), *this);

    host_ptr host = get_memory(count);

    CUresult res;

    set();

    ret->begin(s);
    res = cuMemcpyDtoH(host, get_cuda_ptr(src), count);
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
        s.set_last_event(ret);
    }

    put_memory(host, count);

    return ret;
}

hal_event_ptr 
aspace::copy_async(hal::ptr dst, hal::const_ptr src, size_t count, hal_stream &_s, list_event_detail *_dependencies, gmacError_t &err)
{
    stream &s = reinterpret_cast<stream &>(_s);
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(s);
    }

    event_ptr ret = create_event(true, get_event_type(dst, src), *this);

    CUresult res;

    set();

    ret->begin(s);
    if (is_device_ptr(dst) &&
        is_device_ptr(src)) {
        TRACE(LOCAL, "D (%p) -> D (%p) async copy ("FMT_SIZE" bytes) on "FMT_ID2,
                     get_cuda_ptr(src),
                     get_cuda_ptr(dst),
                     count, s.get_print_id2());

        if (has_direct_copy(src, dst)) {
            res = cuMemcpyDtoDAsync(get_cuda_ptr(dst),
                                    get_cuda_ptr(src), count, s());
        } else {
            hal_buffer *buffer = get_input_buffer(count, s, ret);

            res = cuMemcpyDtoHAsync(buffer->get_addr(), get_cuda_ptr(src), count, s());
            if (res == CUDA_SUCCESS) {
                // TODO, check if an explicit synchronization is required
                res = cuMemcpyHtoDAsync(get_cuda_ptr(dst), buffer->get_addr(), count, s());
            }
        }
    } else if (is_device_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> D (%p) async copy ("FMT_SIZE" bytes) on "FMT_ID2,
                     get_host_ptr(src),
                     get_cuda_ptr(dst),
                     count, s.get_print_id2());
        hal_buffer *buffer = get_output_buffer(count, s, ret);

        memcpy(buffer->get_addr(), get_host_ptr(src), count);

        res = cuMemcpyHtoDAsync(get_cuda_ptr(dst), get_host_ptr(src), count, s());
    } else if (is_host_ptr(dst) &&
               is_device_ptr(src)) {
        TRACE(LOCAL, "D (%p) -> H (%p) async copy ("FMT_SIZE" bytes) on "FMT_ID2,
                     get_cuda_ptr(src),
                     get_host_ptr(dst),
                     count, s.get_print_id2());
        hal_event_ptr last = s.get_last_event();
        last->sync();

        hal_buffer *buffer = get_input_buffer(count, s, ret);

        res = cuMemcpyDtoHAsync(get_host_ptr(dst), get_cuda_ptr(src), count, s());

        // Perform memcpy after asynchronous copy
        ret->add_trigger(do_func(memcpy, buffer->get_addr(), get_host_ptr(src), count));
    } else if (is_host_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy ("FMT_SIZE" bytes) on stream: "FMT_ID2,
                     get_host_ptr(src),
                     get_host_ptr(dst),
                     count, s.get_print_id2());

        res = CUDA_SUCCESS;
        memcpy(get_host_ptr(dst), get_host_ptr(src), count);
    }

    ret->end();

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    } else {
        s.set_last_event(ret);
    }

    return ret;
}

hal_event_ptr
aspace::copy_async(hal::ptr dst, device_input &input, size_t count, hal_stream &_s, list_event_detail *_dependencies, gmacError_t &err)
{
    stream &s = reinterpret_cast<stream &>(_s);
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(s);
    }

    TRACE(LOCAL, "IO -> D async copy ("FMT_SIZE" bytes) on stream: %p", count, s());
    event_ptr ret = create_event(true, get_event_type(dst, input), *this);

    //buffer_t &buffer = stream.get_buffer(count);
    hal_buffer *buffer = get_output_buffer(count, s, ret);

    bool ok = input.read(buffer->get_addr(), count);

    if (ok) {
        CUresult res;

        set();

        ret->begin(s);
        res = cuMemcpyHtoDAsync(get_cuda_ptr(dst), buffer->get_addr(), count, s());
        ret->end();

        err = error(res);
        if (err != gmacSuccess) {
            ret.reset();
        } else {
            s.set_last_event(ret);
            //ret.add_trigger(do_member(stream::put_buffer, &stream, buffer));
        }
    }

    return ret;
}

hal_event_ptr
aspace::copy_async(device_output &output, hal::const_ptr src, size_t count, hal_stream &_s, list_event_detail *_dependencies, gmacError_t &err)
{
    stream &s = reinterpret_cast<stream &>(_s);
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(s);
    }

    TRACE(LOCAL, "D -> IO async copy ("FMT_SIZE" bytes) on stream: %p", count, s());
    event_ptr ret = create_event(true, get_event_type(output, src), *this);

    hal_buffer *buffer = get_input_buffer(count, s, ret);
    //host_ptr host = get_memory(count);

    CUresult res;

    set();

    ret->begin(s);
    res = cuMemcpyDtoHAsync(buffer->get_addr(), get_cuda_ptr(src), count, s());
    ret->end();

    err = ret->sync();
    if (err == gmacSuccess) {
        // TODO: use real async I/O
        bool ok = output.write(buffer->get_addr(), count);

        if (ok) {
            err = error(res);
        } else {
            err = gmacErrorIO;
        }
    }

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        s.set_last_event(ret);
    }

    return ret;
}

hal_event_ptr 
aspace::memset(hal::ptr dst, int c, size_t count, hal_stream &_s, list_event_detail *_dependencies, gmacError_t &err)
{
    stream &s = reinterpret_cast<stream &>(_s);
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(s);
    }

    TRACE(LOCAL, "memset ("FMT_SIZE" bytes) on stream: %p", count, s());
    event_ptr ret = create_event(false, hal_event::type::TransferToDevice, *this);

    set();

    ret->begin(s);
    CUresult res = cuMemsetD8(get_cuda_ptr(dst), (unsigned char)c, count);
    ret->end();

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        s.set_last_event(ret);
    }

    return ret;
}

hal_event_ptr 
aspace::memset_async(hal::ptr dst, int c, size_t count, hal_stream &_s, list_event_detail *_dependencies, gmacError_t &err)
{
    stream &s = reinterpret_cast<stream &>(_s);
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    if (dependencies != NULL) {
        dependencies->set_barrier(s);
    }

    set();

    TRACE(LOCAL, "memset_async ("FMT_SIZE" bytes) on stream: %p", count, s());
    event_ptr ret = create_event(true, hal_event::type::TransferToDevice, *this);

    ret->begin(s);
    CUresult res = cuMemsetD8Async(get_cuda_ptr(dst), (unsigned char)c, count, s());
    ret->end();

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
    } else {
        s.set_last_event(ret);
    }

    return ret;
}

#if 0
hal::ptr
aspace::alloc(size_t count, gmacError_t &err)
{
    set();

    CUdeviceptr devPtr = 0;
    CUresult res = cuMemAlloc(&devPtr, count);

    err = cuda::error(res);

    return hal::ptr(hal::ptr::backend_ptr(devPtr), this);
}
#endif

hal::ptr
aspace::map(hal_object &obj, gmacError_t &err)
{
    set();

    CUdeviceptr devPtr = 0;
    CUresult res = cuMemAlloc(&devPtr, obj.get_size());

    err = cuda::error(res);

    if (err == gmacSuccess) {
        detail::virt::object_view *view = obj.create_view(*this, devPtr, err);
        return hal::ptr(view);
    }

    return hal::ptr();
}

hal::ptr
aspace::map(hal_object &obj, ptrdiff_t offset, gmacError_t &err)
{
    FATAL("Not implementable without driver support");
    return hal::ptr();
}

#if 0
hal::ptr
aspace::alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err)
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

    return hal::ptr(host_ptr(addr), this);
}
#endif

hal_buffer *
aspace::alloc_buffer(size_t size, GmacProtection hint, hal_stream &/*stream*/, gmacError_t &err)
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
aspace::free(hal::ptr acc)
{
    set();

    CUresult ret = cuMemFree(CUdeviceptr(acc.get_view().get_offset()));

    return cuda::error(ret);
}

gmacError_t
aspace::free_buffer(hal_buffer &buffer)
{
    set();

    CUresult ret = cuMemFreeHost(buffer.get_addr());

    return cuda::error(ret);
}

#if 0
gmacError_t
aspace::free_host_pinned(hal::ptr ptr)
{
    set();

    CUresult ret = cuMemFreeHost(ptr.get_host_ptr());

    return cuda::error(ret);
}
#endif

CUcontext &
aspace::operator()()
{
    return context_;
}

const CUcontext &
aspace::operator()() const
{
    return context_;
}

}}}}
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
