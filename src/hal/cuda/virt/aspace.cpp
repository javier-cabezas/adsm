#include "util/misc.h"

#include "hal/cuda/types.h"

#include "hal/cuda/phys/platform.h"
#include "hal/cuda/phys/processing_unit.h"

namespace __impl { namespace hal { namespace cuda { namespace virt {

typedef std::function<CUresult()> op_functor;

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

aspace::aspace(hal_aspace::set_processing_unit &compatibleUnits, phys::aspace &pas, gmacError_t &err) :
    parent(compatibleUnits, pas, err)
{
    TRACE(LOCAL, FMT_ID2" Created", get_print_id2());

    phys::processing_unit &pu = reinterpret_cast<phys::processing_unit &>(**compatibleUnits.begin());

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
    
    CUstream streamToHost, streamToDevice, streamCompute;
    res = cuStreamCreate(&streamToHost, 0);
    if (res != CUDA_SUCCESS) {
        err = error(res);
        return;
    }
    res = cuStreamCreate(&streamToDevice, 0);
    if (res != CUDA_SUCCESS) {
        err = error(res);
        return;
    }
    res = cuStreamCreate(&streamCompute, 0);
    if (res != CUDA_SUCCESS) {
        err = error(res);
        return;
    }

    streamToHost_    = new stream(*this, streamToHost);
    streamToDevice_  = new stream(*this, streamToDevice);
    streamCompute_   = new stream(*this, streamCompute);
    // Use the same stream for intra-device copies and compute
    streamDevice_    = streamCompute_;

    res = cuCtxPopCurrent(&tmp);
    if (res != CUDA_SUCCESS) {
        err = error(res);
        return;
    }

    context_ = ctx;
}

aspace::~aspace()
{
    delete streamToHost_;
    delete streamToDevice_;
    delete streamCompute_;
    // Use the same stream for intra-device copies and compute
}

#if 0
static
code::map_context_repository Modules_("map_context_repository");
#endif

hal_code_repository_view *
aspace::map(const hal_code_repository &repo, gmacError_t &err)
{
    code::repository_view *ret = new code::repository_view(*this, repo, err);

    return ret;
}

gmacError_t
aspace::unmap(hal_code_repository_view &view)
{
    delete &view;

    return gmacSuccess;
}

#if 0
hal_code_repository &
aspace::get_code_repository()
{
    code::repository *repository;
    code::map_context_repository::iterator it = Modules_.find(this);
    if (it == Modules_.end()) {
        set();

        repository = code::module_descriptor::create_modules();
        Modules_.insert(code::map_context_repository::value_type(this, repository));
    } else {
        repository = it->second;
    }

    return *repository;
}
#endif

hal_event_ptr 
aspace::copy(hal::ptr dst, hal::const_ptr src, size_t count, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

#if 0
    if (dependencies != NULL) {
        s->set_barrier(*dependencies);
    }
#endif

    set();

    event_ptr ret = create_event(false, get_event_type(dst, src), *this);

    CUresult res;

    //ret->begin(streamDevice_);
    if (is_device_ptr(dst) &&
        is_device_ptr(src)) {

        if (has_direct_copy(src, dst)) {
            TRACE(LOCAL, "D (%p) -> D (%p) copy (" FMT_SIZE" bytes) on " FMT_ID2, get_cuda_ptr(src),
                                                                                  get_cuda_ptr(dst), count,
                                                                                  streamDevice_->get_print_id2());

            // Wait for dependencies
            if (dependencies != NULL) streamDevice_->set_barrier(*dependencies);

            auto op = [&]() -> CUresult
                      {
                          return cuMemcpyDtoDAsync(get_cuda_ptr(dst),
                                                   get_cuda_ptr(src), count, (*streamDevice_)());
                      };

            res = ret->add_operation(ret, *streamDevice_, op_functor(std::cref(op)));
        } else {
            TRACE(LOCAL, "D (%p) -> H -> D (%p) copy (" FMT_SIZE" bytes) on " FMT_ID2" + " FMT_ID2,
                                                                           get_cuda_ptr(src),
                                                                           get_cuda_ptr(dst), count,
                                                                           streamToHost_->get_print_id2(),
                                                                           streamToDevice_->get_print_id2());

            // TODO: deprecate this path?
            // Wait for dependencies
            if (dependencies != NULL) streamToHost_->set_barrier(*dependencies);

            host_ptr host = get_memory(count);

            auto op1 = [&]() -> CUresult
                       {
                           return cuMemcpyDtoHAsync(host, get_cuda_ptr(src), count, (*streamToHost_)());
                       };

            auto op2 = [&]() -> CUresult
                       {
                           return cuMemcpyHtoDAsync(get_cuda_ptr(dst), host, count, (*streamToDevice_)());
                       };

            res = ret->add_operation(ret, *streamToHost_, op_functor(std::cref(op1)));
            // Wait for the first copy
            if (res == CUDA_SUCCESS && ret->sync_no_exec() == gmacSuccess) {
                res = ret->add_operation(ret, *streamToDevice_, op_functor(std::cref(op2)));
            }

            put_memory(host, count);
        }
    } else if (is_device_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> D (%p) copy (" FMT_SIZE" bytes) on " FMT_ID2, get_host_ptr(src),
                                                                            get_cuda_ptr(dst),
                                                                            count, streamToDevice_->get_print_id2());
        // Wait for dependencies
        if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

        auto op = [&]() -> CUresult
                  {
                      return cuMemcpyHtoDAsync(get_cuda_ptr(dst), get_host_ptr(src), count, (*streamToDevice_)());
                  };
        res = ret->add_operation(ret, *streamToDevice_, op_functor(std::cref(op)));
    } else if (is_host_ptr(dst) &&
               is_device_ptr(src)) {
        TRACE(LOCAL, "D -> H (%p) copy (" FMT_SIZE" bytes) on " FMT_ID2, get_cuda_ptr(src),
                                                                       get_host_ptr(dst),
                                                                       count, streamToHost_->get_print_id2());
        // Wait for dependencies
        if (dependencies != NULL) streamToHost_->set_barrier(*dependencies);

        auto op = [&]() -> CUresult
                  {
                      return cuMemcpyDtoHAsync(get_host_ptr(dst), get_cuda_ptr(src), count, (*streamToHost_)());
                  };
        res = ret->add_operation(ret, *streamToHost_, op_functor(std::cref(op)));
    } else if (is_host_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy (" FMT_SIZE" bytes)", get_host_ptr(src), get_host_ptr(dst), count);

        res = CUDA_SUCCESS;
        ::memcpy(get_host_ptr(dst), get_host_ptr(src), count);
    } else {
        FATAL("Unhandled case");
    }

    err = error(res);
    // Wait for the copy to complete, before return
    if (err != gmacSuccess || ((err = ret->sync()) != gmacSuccess)) {
        ret.reset();
    }
    return ret;
}

hal_event_ptr
aspace::copy(hal::ptr dst, device_input &input, size_t count, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "IO -> D copy (" FMT_SIZE" bytes) on " FMT_ID2, count, streamToDevice_->get_print_id2());
    event_ptr ret = create_event(false, get_event_type(dst, input), *this);

    host_ptr host = get_memory(count);

    bool ok = input.read(host, count);

    if (ok) {
        CUresult res;

        set();

        // Wait for dependencies
        if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

        auto op = [&]() -> CUresult
                  {
                      return cuMemcpyHtoDAsync(get_cuda_ptr(dst), host, count, (*streamToDevice_)());
                  };

        res = ret->add_operation(ret, *streamToDevice_, op_functor(std::cref(op)));

        err = error(res);
        // Wait for the copy to complete, before return
        if (err != gmacSuccess || ((err = ret->sync()) != gmacSuccess)) {
            ret.reset();
        }
    }

    put_memory(host, count);

    return ret;
}

hal_event_ptr
aspace::copy(device_output &output, hal::const_ptr src, size_t count, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "D -> IO copy (" FMT_SIZE" bytes) on " FMT_ID2, count, streamToHost_->get_print_id2());
    event_ptr ret = create_event(false, get_event_type(output, src), *this);

    host_ptr host = get_memory(count);

    // Wait for dependencies
    if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

    CUresult res;

    set();

    auto op = [&]() -> CUresult
              {
                  return cuMemcpyDtoHAsync(host, get_cuda_ptr(src), count, (*streamToHost_)());
              };

    res = ret->add_operation(ret, *streamToHost_, op_functor(std::cref(op)));
    err = error(res);

    if (err == gmacSuccess) {
        err = ret->sync();

        if (err == gmacSuccess) {
            bool ok = output.write(host, count);

            if (!ok) {
                err = gmacErrorIO;
            }
        }
    }

    if (err != gmacSuccess) {
        ret.reset();
    }

    put_memory(host, count);

    return ret;
}

hal_event_ptr 
aspace::copy_async(hal::ptr dst, hal::const_ptr src, size_t count, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    event_ptr ret = create_event(true, get_event_type(dst, src), *this);

    CUresult res;

    set();

    if (is_device_ptr(dst) &&
        is_device_ptr(src)) {

        if (has_direct_copy(src, dst)) {
            TRACE(LOCAL, "D -> D copy (" FMT_SIZE" bytes) on " FMT_ID2, get_cuda_ptr(src),
                                                                      get_cuda_ptr(dst), count,
                                                                      streamDevice_->get_print_id2());

            // Wait for dependencies
            if (dependencies != NULL) streamDevice_->set_barrier(*dependencies);

            auto op = [&]() -> CUresult
                      {
                          return cuMemcpyDtoDAsync(get_cuda_ptr(dst),
                                                   get_cuda_ptr(src), count, (*streamDevice_)());
                      };

            res = ret->add_operation(ret, *streamDevice_, op_functor(std::cref(op)));
        } else {
            TRACE(LOCAL, "D (%p) -> H -> D (%p) copy (" FMT_SIZE" bytes) on " FMT_ID2" + " FMT_ID2,
                                                                           get_cuda_ptr(src),
                                                                           get_cuda_ptr(dst), count,
                                                                           streamToHost_->get_print_id2(),
                                                                           streamToDevice_->get_print_id2());

            // TODO: deprecate this path?
            if (dependencies != NULL) streamToHost_->set_barrier(*dependencies);

            // TODO: remove stream parameter
            hal_buffer *buffer = get_input_buffer(count, *streamToHost_, ret);

            auto op1 = [&]() -> CUresult
                       {
                           return cuMemcpyDtoHAsync(buffer->get_addr(), get_cuda_ptr(src), count, (*streamToHost_)());
                       };

            auto op2 = [&]() -> CUresult
                       {
                           return cuMemcpyHtoDAsync(get_cuda_ptr(dst), buffer->get_addr(), count, (*streamToDevice_)());
                       };


            res = ret->add_operation(ret, *streamToHost_, op_functor(std::cref(op1)));
            // Add barrier to avoid data races
            streamToDevice_->set_barrier(ret);
            if (res == CUDA_SUCCESS) {
                res = ret->add_operation(ret, *streamToDevice_, op_functor(std::cref(op2)));
            }
        }
    } else if (is_device_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> D (%p) async copy (" FMT_SIZE" bytes) on " FMT_ID2, get_host_ptr(src),
                                                                         get_cuda_ptr(dst),
                                                                         count, streamToDevice_->get_print_id2());
        // Wait for dependencies
        if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

        hal_buffer *buffer = get_output_buffer(count, *streamToDevice_, ret);

        ::memcpy(buffer->get_addr(), get_host_ptr(src), count);

        auto op = [&]() -> CUresult
                  {
                      return cuMemcpyHtoDAsync(get_cuda_ptr(dst), buffer->get_addr(), count, (*streamToDevice_)());
                  };
        res = ret->add_operation(ret, *streamToDevice_, op_functor(std::cref(op)));
    } else if (is_host_ptr(dst) &&
               is_device_ptr(src)) {
        TRACE(LOCAL, "D (%p) -> H (%p) async copy (" FMT_SIZE" bytes) on " FMT_ID2,
                     get_cuda_ptr(src),
                     get_host_ptr(dst),
                     count, streamToHost_->get_print_id2());
        // Wait for dependencies
        if (dependencies != NULL) streamToHost_->set_barrier(*dependencies);

        hal_buffer *buffer = get_input_buffer(count, *streamToHost_, ret);

        auto op = [&]() -> CUresult
                  {
                      return cuMemcpyDtoHAsync(buffer->get_addr(), get_cuda_ptr(src), count, (*streamToHost_)());
                  };

        // Perform memcpy after asynchronous copy
        ret->add_trigger(do_func(::memcpy, buffer->get_addr(), get_host_ptr(src), count));

        res = ret->add_operation(ret, *streamToHost_, op_functor(std::cref(op)));
    } else if (is_host_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy (" FMT_SIZE" bytes)",
                     get_host_ptr(src),
                     get_host_ptr(dst), count);

        res = CUDA_SUCCESS;
        ::memcpy(get_host_ptr(dst), get_host_ptr(src), count);
    }

    err = error(res);
    if (err != gmacSuccess) {
        ret.reset();
    }

    return ret;
}

hal_event_ptr
aspace::copy_async(hal::ptr dst, device_input &input, size_t count, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "IO -> D async copy (" FMT_SIZE" bytes) on " FMT_ID2, count, streamToDevice_->get_print_id2());
    event_ptr ret = create_event(true, get_event_type(dst, input), *this);

    //buffer_t &buffer = stream.get_buffer(count);
    hal_buffer *buffer = get_output_buffer(count, *streamToDevice_, ret);

    bool ok = input.read(buffer->get_addr(), count);

    if (ok) {
        CUresult res;

        set();

        // Wait for dependencies
        if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

        auto op = [&]() -> CUresult
                  {
                      return cuMemcpyHtoDAsync(get_cuda_ptr(dst), buffer->get_addr(), count, (*streamToDevice_)());
                  };

        res = ret->add_operation(ret, *streamToDevice_, op_functor(std::cref(op)));

        err = error(res);
        if (err != gmacSuccess) {
            ret.reset();
        }
    }

    return ret;
}

hal_event_ptr
aspace::copy_async(device_output &output, hal::const_ptr src, size_t count, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "D -> IO async copy (" FMT_SIZE" bytes) on " FMT_ID2, count, streamToHost_->get_print_id2());
    event_ptr ret = create_event(true, get_event_type(output, src), *this);

    hal_buffer *buffer = get_input_buffer(count, *streamToHost_, ret);

    // Wait for dependencies
    if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

    CUresult res;

    set();

    auto op = [&]() -> CUresult
              {
                  return cuMemcpyDtoHAsync(buffer->get_addr(), get_cuda_ptr(src), count, (*streamToHost_)());
              };

    res = ret->add_operation(ret, *streamToHost_, op_functor(std::cref(op)));
    err = error(res);

    if (err == gmacSuccess) {
        err = ret->sync();

        if (err == gmacSuccess) {
            // TODO: use real async I/O
            bool ok = output.write(buffer->get_addr(), count);

            if (!ok) {
                err = gmacErrorIO;
            }
        }
    }

    if (err != gmacSuccess) {
        ret.reset();
    }

    return ret;
}

hal_event_ptr 
aspace::memset(hal::ptr dst, int c, size_t count, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "memset (" FMT_SIZE" bytes) on stream: %p", count, streamDevice_->get_print_id2());
    event_ptr ret = create_event(false, hal_event::type::TransferToDevice, *this);

    // Wait for dependencies
    if (dependencies != NULL) streamDevice_->set_barrier(*dependencies);

    set();

    auto op = [&]() -> CUresult
              {
                  return cuMemsetD8Async(get_cuda_ptr(dst), (unsigned char)c, count, (*streamDevice_)());
              };

    CUresult res = ret->add_operation(ret, *streamDevice_, op_functor(std::cref(op)));

    err = error(res);

    // Wait for memset to complete, before return
    if (err != gmacSuccess || ((err = ret->sync()) != gmacSuccess)) {
        ret.reset();
    }

    return ret;
}

hal_event_ptr 
aspace::memset_async(hal::ptr dst, int c, size_t count, list_event_detail *_dependencies, gmacError_t &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "memset (" FMT_SIZE" bytes) on stream: %p", count, streamDevice_->get_print_id2());
    event_ptr ret = create_event(false, hal_event::type::TransferToDevice, *this);

    // Wait for dependencies
    if (dependencies != NULL) streamDevice_->set_barrier(*dependencies);

    set();

    auto op = [&]() -> CUresult
              {
                  return cuMemsetD8Async(get_cuda_ptr(dst), (unsigned char)c, count, (*streamDevice_)());
              };

    CUresult res = ret->add_operation(ret, *streamDevice_, op_functor(std::cref(op)));

    err = error(res);

    if (err != gmacSuccess) {
        ret.reset();
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
    if (get_paspace().get_memories().find(&obj.get_memory()) == get_paspace().get_memories().end()) {
        // The object resides in a memory not accessible by this aspace
        err = gmacErrorInvalidValue;
        return hal::ptr();
    }

    if (obj.get_view(*this) != NULL) {
        // Mapping the same object more than once per address space is not supported
        err = gmacErrorFeatureNotSupported;
        return hal::ptr();
    }

    set();

    // Refine this logic
    if (obj.get_views().size() > 0) {
        // CASE 1: Mappings already exist for this object
            
        hal_object::set_view viewsCpu = obj.get_views(phys::hal_processing_unit::PUNIT_TYPE_CPU);

        if (viewsCpu.size() > 0) {
            // CASE 1a: Use host-mapped memory (register)
            void *ptr = (void *) (*viewsCpu.begin())->get_offset();
            CUresult res = cuMemHostRegister(ptr, obj.get_size(), CU_MEMHOSTREGISTER_PORTABLE);

            err = cuda::error(res);

            if (err == gmacSuccess) {
                detail::virt::object_view *view = obj.create_view(*this, hal::ptr::offset_type(ptr), err);
                if (err == gmacSuccess) {
                    return hal::ptr(*view);
                }
            }
        } else {
            hal_object::set_view viewsGpu = obj.get_views(phys::hal_processing_unit::PUNIT_TYPE_GPU);

            if (viewsGpu.size() > 0) {
                // CASE 1b: Use peer access
                const cuda::virt::aspace &vas = reinterpret_cast<const cuda::virt::aspace &>((*viewsGpu.begin())->get_vaspace());
                if ((vas.pUnits_.size() == 1 && pUnits_.size() == 1) &&
                    (*vas.pUnits_.begin() != *pUnits_.begin())) {
                    CUresult res = cuCtxEnablePeerAccess(vas.context_, 0);
                    err = cuda::error(res);

                    if (err != gmacSuccess) {
                        return hal::ptr();
                    }

                    void *ptr = (void *) (*viewsGpu.begin())->get_offset();

                    detail::virt::object_view *view = obj.create_view(*this, hal::ptr::offset_type(ptr), err);

                    if (err == gmacSuccess) {
                        return hal::ptr(*view);
                    }
                } else {
                    // Mapping the same object on address spaces of the same device is not supported
                    err = gmacErrorFeatureNotSupported;
                    return hal::ptr();
                }
            } else {
                FATAL("Unhandled case");
            }
        }
    } else {
        // CASE 2: Mappings do not exist for this object: create one
        CUdeviceptr devPtr = 0;
        CUresult res = cuMemAlloc(&devPtr, obj.get_size());

        err = cuda::error(res);

        if (err == gmacSuccess) {
            detail::virt::object_view *view = obj.create_view(*this, devPtr, err);

            if (err == gmacSuccess) {
                return hal::ptr(*view);
            }
        }
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

    TRACE(LOCAL, "Created buffer: %p (" FMT_SIZE")", addr, size);
    buffer_t *ret = NULL;
    if (res == CUDA_SUCCESS) {
        ret = new buffer_t(host_ptr(addr), size, *this);
    }

    return ret;
}

gmacError_t
aspace::unmap(hal::ptr p)
{
    CUdeviceptr ptr = CUdeviceptr(p.get_view().get_offset());
    hal_object &obj = p.get_view().get_object();
    gmacError_t ret = obj.destroy_view(p.get_view());

    if (ret == gmacSuccess && obj.get_views().size() == 0) {
        // TODO: set the proper AS to destroy on the original device
        // TODO: modify unit test in manager.cpp accordingly
        set();
    
        CUresult err = cuMemFree(CUdeviceptr(ptr));
        ret = cuda::error(err);
    }

    return ret;
}

#if 0
gmacError_t
aspace::free(hal::ptr acc)
{
    set();

    CUresult ret = cuMemFree(CUdeviceptr(acc.get_view().get_offset()));

    return cuda::error(ret);
}
#endif

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
