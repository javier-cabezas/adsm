#include "util/misc.h"

#include "hal/cuda/types.h"

#include "hal/cuda/phys/platform.h"
#include "hal/cuda/phys/processing_unit.h"

namespace __impl { namespace hal { namespace cuda { namespace virt {

const unsigned &aspace::MaxBuffersIn_  = config::params::HALInputBuffersPerContext;
const unsigned &aspace::MaxBuffersOut_ = config::params::HALOutputBuffersPerContext;

buffer *
aspace::get_input_buffer(size_t size, stream &stream, event_ptr event)
{
    buffer *buf = mapFreeBuffersIn_.pop(size);

    if (buf == NULL) {
        nBuffersIn_.lock();
        if (nBuffersIn_ < MaxBuffersIn_) {
            nBuffersIn_++;
            nBuffersIn_.unlock();

            hal::error err;
            buf = alloc_buffer(size, GMAC_PROT_READ, stream, err);
            ASSERTION(err == HAL_SUCCESS);
        } else {
            nBuffersIn_.unlock();
            buf = mapUsedBuffersIn_.pop(size);
            buf->wait();
        }
    } else {
        TRACE(LOCAL, "Reusing input buffer");
    }

    buf->set_event(event);

    mapUsedBuffersIn_.push(buf, buf->get_size());

    return buf;
}

buffer *
aspace::get_output_buffer(size_t size, stream &stream, event_ptr event)
{
    buffer *buf = mapFreeBuffersOut_.pop(size);

    if (buf == NULL) {
        if (nBuffersOut_ < MaxBuffersOut_) {
            hal::error err;

            buf = alloc_buffer(size, GMAC_PROT_WRITE, stream, err);
            ASSERTION(err == HAL_SUCCESS);
            nBuffersOut_++;
        } else {
            buf = mapUsedBuffersOut_.pop(size);
            buf->wait();
        }
    } else {
        TRACE(LOCAL, "Reusing output buffer");
    }

    buf->set_event(event);

    mapUsedBuffersOut_.push(buf, buf->get_size());

    return buf;
}



template <typename Ptr>
static bool
is_host_ptr(Ptr &&p)
{
    detail::virt::object &o = p.get_view().get_object();

    const detail::phys::memory &m = o.get_memory();

    return util::algo::has_predicate(m.get_attached_units(),
                                     [](const detail::phys::processing_unit *pu) -> bool
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

#if 0
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
#endif

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

aspace::aspace(hal_aspace::set_processing_unit &compatibleUnits, phys::aspace &pas, hal::error &err) :
    parent(compatibleUnits, pas, err),
    nBuffersIn_(0),
    nBuffersOut_(0)
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

    context_ = ctx;
    
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
}

aspace::~aspace()
{
    delete streamToHost_;
    delete streamToDevice_;
    delete streamCompute_;
    // Use the same stream for intra-device copies and compute
    
    CUresult ret = cuCtxDestroy(context_);
    ASSERTION(ret == CUDA_SUCCESS);
}

#if 0
static
code::map_context_repository Modules_("map_context_repository");
#endif

hal_code_repository_view *
aspace::map(const hal_code_repository &repo, hal::error &err)
{
    code::repository_view *ret = new code::repository_view(*this, repo, err);

    return ret;
}

hal::error
aspace::unmap(hal_code_repository_view &view)
{
    delete &view;

    return HAL_SUCCESS;
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
aspace::copy(hal::ptr dst, hal::const_ptr src, size_t count, list_event_detail *_dependencies, hal::error &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    set();

    event_ptr ret = create_event(false, cuda::_event_t::Memory, *this);

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

            auto op = [&](CUstream s) -> CUresult
                      {
                          return cuMemcpyDtoDAsync(get_cuda_ptr(dst),
                                                   get_cuda_ptr(src), count, s);
                      };

            res = ret->add_operation(ret, *streamDevice_, operation::func_op(std::cref(op)), operation::TransferDevice, false);
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

            auto op1 = [&](CUstream s) -> CUresult
                       {
                           return cuMemcpyDtoHAsync(host, get_cuda_ptr(src), count, s);
                       };

            auto op2 = [&](CUstream s) -> CUresult
                       {
                           return cuMemcpyHtoDAsync(get_cuda_ptr(dst), host, count, s);
                       };

            res = ret->add_operation(ret, *streamToHost_, operation::func_op(std::cref(op1)), operation::TransferToHost, false);
            // Wait for the first copy
            if (res == CUDA_SUCCESS) {
                res = ret->add_operation(ret, *streamToDevice_, operation::func_op(std::cref(op2)), operation::TransferToDevice, false);
            }

            put_memory(host, count);
        }
    } else if (is_device_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> D (%p) copy (" FMT_SIZE" bytes) on " FMT_ID2,
                     get_host_ptr(src),
                     get_cuda_ptr(dst),
                     count,
                     streamToDevice_->get_print_id2());
        // Wait for dependencies
        if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

        auto op = [&](CUstream s) -> CUresult
                  {
                      return cuMemcpyHtoDAsync(get_cuda_ptr(dst), get_host_ptr(src), count, s);
                  };
        res = ret->add_operation(ret, *streamToDevice_, operation::func_op(std::cref(op)), operation::TransferToDevice, false);
    } else if (is_host_ptr(dst) &&
               is_device_ptr(src)) {
        TRACE(LOCAL, "D -> H (%p) copy (" FMT_SIZE" bytes) on " FMT_ID2, get_cuda_ptr(src),
                                                                       get_host_ptr(dst),
                                                                       count, streamToHost_->get_print_id2());
        // Wait for dependencies
        if (dependencies != NULL) streamToHost_->set_barrier(*dependencies);

        auto op = [&](CUstream s) -> CUresult
                  {
                      return cuMemcpyDtoHAsync(get_host_ptr(dst), get_cuda_ptr(src), count, s);
                  };
        res = ret->add_operation(ret, *streamToHost_, operation::func_op(std::cref(op)), operation::TransferToHost, false);
    } else if (is_host_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy (" FMT_SIZE" bytes)", get_host_ptr(src), get_host_ptr(dst), count);

        res = CUDA_SUCCESS;
        auto op = [&](CUstream) -> CUresult
                  {
                      ::memcpy(get_host_ptr(dst), get_host_ptr(src), count);
                      return CUDA_SUCCESS;
                  };
        res = ret->add_operation(ret, *streamToHost_, operation::func_op(std::cref(op)), operation::TransferHost, false);
    } else {
        FATAL("Unhandled case");
    }

    err = error(res);
    // Wait for the copy to complete, before return
    if (err != HAL_SUCCESS || ((err = ret->sync()) != HAL_SUCCESS)) {
        ret.reset();
    }
    return ret;
}

hal_event_ptr
aspace::copy(hal::ptr dst, device_input &input, size_t count, list_event_detail *_dependencies, hal::error &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "IO -> D copy (" FMT_SIZE" bytes) on " FMT_ID2, count, streamToDevice_->get_print_id2());
    event_ptr ret = create_event(false, _event_t::IO, *this);

    host_ptr host = get_memory(count);

    bool ok = input.read(host, count);

    if (ok) {
        CUresult res;

        set();

        // Wait for dependencies
        if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

        auto op = [&](CUstream s) -> CUresult
                  {
                      return cuMemcpyHtoDAsync(get_cuda_ptr(dst), host, count, s);
                  };

        res = ret->add_operation(ret, *streamToDevice_, operation::func_op(std::cref(op)), operation::TransferToDevice, true);

        err = error(res);
        // Wait for the copy to complete, before return
        if (err != HAL_SUCCESS || ((err = ret->sync()) != HAL_SUCCESS)) {
            ret.reset();
        }
    }

    put_memory(host, count);

    return ret;
}

hal_event_ptr
aspace::copy(device_output &output, hal::const_ptr src, size_t count, list_event_detail *_dependencies, hal::error &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "D -> IO copy (" FMT_SIZE" bytes) on " FMT_ID2, count, streamToHost_->get_print_id2());
    event_ptr ret = create_event(false, _event_t::IO, *this);

    host_ptr host = get_memory(count);

    // Wait for dependencies
    if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

    CUresult res;

    set();

    auto op = [&](CUstream s) -> CUresult
              {
                  return cuMemcpyDtoHAsync(host, get_cuda_ptr(src), count, s);
              };

    res = ret->add_operation(ret, *streamToHost_, operation::func_op(std::cref(op)), operation::TransferToHost, true);
    err = error(res);

    if (err == HAL_SUCCESS) {
        err = ret->sync();

        if (err == HAL_SUCCESS) {
            bool ok = output.write(host, count);

            if (!ok) {
                err = HAL_ERROR_IO;
            }
        }
    }

    if (err != HAL_SUCCESS) {
        ret.reset();
    }

    put_memory(host, count);

    return ret;
}

hal_event_ptr 
aspace::copy_async(hal::ptr dst, hal::const_ptr src, size_t count, list_event_detail *_dependencies, hal::error &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    event_ptr ret = create_event(true, _event_t::Memory, *this);

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

            auto op = [&](CUstream s) -> CUresult
                      {
                          return cuMemcpyDtoDAsync(get_cuda_ptr(dst),
                                                   get_cuda_ptr(src), count, s);
                      };

            res = ret->add_operation(ret, *streamDevice_, operation::func_op(std::cref(op)), operation::TransferDevice, true);
        } else {
            TRACE(LOCAL, "D (%p) -> H -> D (%p) copy (" FMT_SIZE" bytes) on " FMT_ID2" + " FMT_ID2,
                         get_cuda_ptr(src),
                         get_cuda_ptr(dst), count,
                         streamToHost_->get_print_id2(),
                         streamToDevice_->get_print_id2());

            // TODO: deprecate this path?
            if (dependencies != NULL) streamToHost_->set_barrier(*dependencies);

            // TODO: remove stream parameter
            buffer *buf = get_input_buffer(count, *streamToHost_, ret);

            auto op1 = [&](CUstream s) -> CUresult
                       {
                           return cuMemcpyDtoHAsync(buf->get_addr(), get_cuda_ptr(src), count, s);
                       };

            auto op2 = [&](CUstream s) -> CUresult
                       {
                           return cuMemcpyHtoDAsync(get_cuda_ptr(dst), buf->get_addr(), count, s);
                       };


            res = ret->add_operation(ret, *streamToHost_, operation::func_op(std::cref(op1)), operation::TransferToHost, true);
            // Add barrier to avoid data races
            streamToDevice_->set_barrier(*ret);
            if (res == CUDA_SUCCESS) {
                res = ret->add_operation(ret, *streamToDevice_, operation::func_op(std::cref(op2)), operation::TransferToDevice, true);
            }
        }
    } else if (is_device_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> D (%p) async copy (" FMT_SIZE" bytes) on " FMT_ID2,
                     get_host_ptr(src),
                     get_cuda_ptr(dst),
                     count, streamToDevice_->get_print_id2());
        // Wait for dependencies
        if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

        buffer *buf = get_output_buffer(count, *streamToDevice_, ret);


        auto op1 = [&](CUstream /*s*/) -> CUresult
                   {
                       ::memcpy(buf->get_addr(), get_host_ptr(src), count);
                       return CUDA_SUCCESS;
                   };

        auto op2 = [&](CUstream s) -> CUresult
                   {
                       return cuMemcpyHtoDAsync(get_cuda_ptr(dst), buf->get_addr(), count, s);
                   };
        res = ret->add_operation(ret, *streamToHost_, operation::func_op(std::cref(op1)), operation::TransferToDevice, false);
        if (res == CUDA_SUCCESS) {
            res = ret->add_operation(ret, *streamToDevice_, operation::func_op(std::cref(op2)), operation::TransferToDevice, true);
        }
    } else if (is_host_ptr(dst) &&
               is_device_ptr(src)) {
        TRACE(LOCAL, "D (%p) -> H (%p) async copy (" FMT_SIZE" bytes) on " FMT_ID2,
                     get_cuda_ptr(src),
                     get_host_ptr(dst),
                     count, streamToHost_->get_print_id2());
        // Wait for dependencies
        if (dependencies != NULL) streamToHost_->set_barrier(*dependencies);

        buffer *buf = get_input_buffer(count, *streamToHost_, ret);

        auto op1 = [&](CUstream s) -> CUresult
                   {
                       return cuMemcpyDtoHAsync(buf->get_addr(), get_cuda_ptr(src), count, s);
                   };

        auto op2 = [=](CUstream) -> CUresult
                   {
                       ::memcpy(buf->get_addr(), get_host_ptr(src), count);
                       return CUDA_SUCCESS;
                   };

#if 0
        // Perform memcpy after asynchronous copy
        ret->add_trigger(do_func(::memcpy, buf->get_addr(), get_host_ptr(src), count));
#endif
        res = ret->add_operation(ret, *streamToHost_, operation::func_op(std::cref(op1)), operation::TransferToHost, true);
        if (res == CUDA_SUCCESS) {
            res = ret->add_operation(ret, *streamToHost_, operation::func_op(std::cref(op2)), operation::TransferHost, false);
        }
    } else if (is_host_ptr(dst) &&
               is_host_ptr(src)) {
        TRACE(LOCAL, "H (%p) -> H (%p) copy (" FMT_SIZE" bytes)",
                     get_host_ptr(src),
                     get_host_ptr(dst), count);

        auto op = [=](CUstream) -> CUresult
                  {
                      ::memcpy(get_host_ptr(dst), get_host_ptr(src), count);
                      return CUDA_SUCCESS;
                  };
        res = CUDA_SUCCESS;
        res = ret->add_operation(ret, *streamToHost_, operation::func_op(std::cref(op)), operation::TransferHost, false);
    }

    err = error(res);
    if (err != HAL_SUCCESS) {
        ret.reset();
    }

    return ret;
}

hal_event_ptr
aspace::copy_async(hal::ptr dst, device_input &input, size_t count, list_event_detail *_dependencies, hal::error &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "IO -> D async copy (" FMT_SIZE" bytes) on " FMT_ID2,
                 count,
                 streamToDevice_->get_print_id2());
    event_ptr ret = create_event(true, _event_t::IO, *this);

    //buffer_t &buffer = stream.get_buffer(count);
    buffer *buf = get_output_buffer(count, *streamToDevice_, ret);

    bool ok = input.read(buf->get_addr(), count);

    if (ok) {
        CUresult res;

        set();

        // Wait for dependencies
        if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

        auto op = [&](CUstream s) -> CUresult
                  {
                      return cuMemcpyHtoDAsync(get_cuda_ptr(dst), buf->get_addr(), count, s);
                  };

        res = ret->add_operation(ret, *streamToDevice_, operation::func_op(std::cref(op)), operation::TransferToDevice, true);

        err = error(res);
        if (err != HAL_SUCCESS) {
            ret.reset();
        }
    }

    return ret;
}

hal_event_ptr
aspace::copy_async(device_output &output, hal::const_ptr src, size_t count, list_event_detail *_dependencies, hal::error &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "D -> IO async copy (" FMT_SIZE" bytes) on " FMT_ID2,
                 count,
                 streamToHost_->get_print_id2());
    event_ptr ret = create_event(true, _event_t::IO, *this);

    buffer *buf = get_input_buffer(count, *streamToHost_, ret);

    // Wait for dependencies
    if (dependencies != NULL) streamToDevice_->set_barrier(*dependencies);

    CUresult res;

    set();

    auto op = [&](CUstream s) -> CUresult
              {
                  return cuMemcpyDtoHAsync(buf->get_addr(), get_cuda_ptr(src), count, s);
              };

    res = ret->add_operation(ret, *streamToHost_, operation::func_op(std::cref(op)), operation::TransferToHost, true);
    err = error(res);

    if (err == HAL_SUCCESS) {
        err = ret->sync();

        if (err == HAL_SUCCESS) {
            // TODO: use real async I/O
            bool ok = output.write(buf->get_addr(), count);

            if (!ok) {
                err = HAL_ERROR_IO;
            }
        }
    }

    if (err != HAL_SUCCESS) {
        ret.reset();
    }

    return ret;
}

hal_event_ptr 
aspace::memset(hal::ptr dst, int c, size_t count, list_event_detail *_dependencies, hal::error &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "memset (" FMT_SIZE" bytes) on stream: "FMT_ID2,
                 count,
                 streamDevice_->get_print_id2());
    event_ptr ret = create_event(false, hal_event::Memory, *this);

    // Wait for dependencies
    if (dependencies != NULL) streamDevice_->set_barrier(*dependencies);

    set();

    auto op = [&](CUstream s) -> CUresult
              {
                  return cuMemsetD8Async(get_cuda_ptr(dst), (unsigned char)c, count, s);
              };

    CUresult res = ret->add_operation(ret, *streamDevice_, operation::func_op(std::cref(op)), operation::TransferDevice, true);

    err = error(res);

    // Wait for memset to complete, before return
    if (err != HAL_SUCCESS || ((err = ret->sync()) != HAL_SUCCESS)) {
        ret.reset();
    }

    return ret;
}

hal_event_ptr 
aspace::memset_async(hal::ptr dst, int c, size_t count, list_event_detail *_dependencies, hal::error &err)
{
    list_event *dependencies = reinterpret_cast<list_event *>(_dependencies);

    TRACE(LOCAL, "memset (" FMT_SIZE" bytes) on stream: "FMT_ID2,
                 count,
                 streamDevice_->get_print_id2());
    event_ptr ret = create_event(false, hal_event::Memory, *this);

    // Wait for dependencies
    if (dependencies != NULL) streamDevice_->set_barrier(*dependencies);

    set();

    auto op = [&](CUstream s) -> CUresult
              {
                  return cuMemsetD8Async(get_cuda_ptr(dst), (unsigned char)c, count, s);
              };

    CUresult res = ret->add_operation(ret, *streamDevice_, operation::func_op(std::cref(op)), operation::TransferDevice, true);

    err = error(res);

    if (err != HAL_SUCCESS) {
        ret.reset();
    }

    return ret;
}

#if 0
hal::ptr
aspace::alloc(size_t count, hal::error &err)
{
    set();

    CUdeviceptr devPtr = 0;
    CUresult res = cuMemAlloc(&devPtr, count);

    err = cuda::error_to_hal(res);

    return hal::ptr(hal::ptr::backend_ptr(devPtr), this);
}
#endif

hal::ptr
aspace::map(hal_object &obj, GmacProtection prot, hal::error &err)
{
    WARNING("Using protection flags in CUDA is not supported");

    if (get_paspace().get_memories().find(&obj.get_memory()) == get_paspace().get_memories().end()) {
        // The object resides in a memory not accessible by this aspace
        err = HAL_ERROR_INVALID_VALUE;
        return hal::ptr();
    }

    if (obj.get_views(*this).size() != 0) {
        // Mapping the same object more than once per address space is not supported
        err = HAL_ERROR_FEATURE_NOT_SUPPORTED;
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

            err = cuda::error_to_hal(res);

            if (err == HAL_SUCCESS) {
                detail::virt::object_view *view = obj.create_view(*this, hal::ptr::offset_type(ptr), err);
                if (err == HAL_SUCCESS) {
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
                    err = cuda::error_to_hal(res);

                    if (err != HAL_SUCCESS) {
                        return hal::ptr();
                    }

                    void *ptr = (void *) (*viewsGpu.begin())->get_offset();

                    detail::virt::object_view *view = obj.create_view(*this, hal::ptr::offset_type(ptr), err);

                    if (err == HAL_SUCCESS) {
                        return hal::ptr(*view);
                    }
                } else {
                    // Mapping the same object on address spaces of the same device is not supported
                    err = HAL_ERROR_FEATURE_NOT_SUPPORTED;
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

        err = cuda::error_to_hal(res);

        if (err == HAL_SUCCESS) {
            detail::virt::object_view *view = obj.create_view(*this, devPtr, err);

            if (err == HAL_SUCCESS) {
                return hal::ptr(*view);
            }
        }
    }

    return hal::ptr();
}

hal::ptr
aspace::map(hal_object &obj, GmacProtection prot, ptrdiff_t offset, hal::error &err)
{
    FATAL("Not implementable without driver support");
    return hal::ptr();
}

#if 0
hal::ptr
aspace::alloc_host_pinned(size_t size, GmacProtection hint, hal::error &err)
{
    set();

    // TODO: add a parater to specify accesibility of the buffer from the device
    unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
    if (hint == GMAC_PROT_WRITE) {
        flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    void *addr;
    CUresult res = cuMemHostAlloc(&addr, size, flags);
    err = cuda::error_to_hal(res);

    return hal::ptr(host_ptr(addr), this);
}
#endif

buffer *
aspace::alloc_buffer(size_t size, GmacProtection hint, stream &/*stream*/, hal::error &err)
{
    set();

    // TODO: add a parater to specify accesibility of the buffer from the device
    unsigned flags = CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP;
    if (hint == GMAC_PROT_WRITE) {
        flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    void *addr;
    CUresult res = cuMemHostAlloc(&addr, size, flags);
    err = cuda::error_to_hal(res);

    TRACE(LOCAL, "Created buffer: %p (" FMT_SIZE")", addr, size);
    buffer *ret = NULL;
    if (res == CUDA_SUCCESS) {
        ret = factory_buffer::create(host_ptr(addr), size, *this);
    }

    return ret;
}

hal::error
aspace::unmap(hal::ptr p)
{
    CUdeviceptr ptr = CUdeviceptr(p.get_view().get_offset());
    hal_object &obj = p.get_view().get_object();
    hal::error ret = obj.destroy_view(p.get_view());

    if (ret == HAL_SUCCESS && obj.get_views().size() == 0) {
        // TODO: set the proper AS to destroy on the original device
        // TODO: modify unit test in manager.cpp accordingly
        set();
    
        CUresult err = cuMemFree(CUdeviceptr(ptr));
        ret = cuda::error_to_hal(err);
    }

    return ret;
}

#if 0
hal::error
aspace::free(hal::ptr acc)
{
    set();

    CUresult ret = cuMemFree(CUdeviceptr(acc.get_view().get_offset()));

    return cuda::error_to_hal(ret);
}
#endif

hal::error
aspace::free_buffer(buffer &buf)
{
    set();

    CUresult ret = cuMemFreeHost(buf.get_addr());

    return cuda::error_to_hal(ret);
}

#if 0
hal::error
aspace::free_host_pinned(hal::ptr ptr)
{
    set();

    CUresult ret = cuMemFreeHost(ptr.get_host_ptr());

    return cuda::error_to_hal(ret);
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
