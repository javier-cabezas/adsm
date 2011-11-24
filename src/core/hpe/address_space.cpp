#include "address_space.h"
#include "thread.h"
#include "vdevice.h"

namespace __impl { namespace core { namespace hpe {

#if 0
context &
address_space::get_context()
{
    context *context = thread::get_current_thread().get_context(*this);
    if (context == NULL) {
        context = proc_.get_resource_manager().create_context(util::GetThreadId(), *this);
        ASSERTION(context != NULL);
    }

    return *context;
}
#endif

address_space::address_space(hal::context_t &context,
                             hal::stream_t &streamLaunch,
                             hal::stream_t &streamToAccelerator,
                             hal::stream_t &streamToHost,
                             hal::stream_t &streamAccelerator,
                             process &proc) :
    Lock("address_space"),
    ctx_(context),
    streamLaunch_(streamLaunch),
    streamToAccelerator_(streamToAccelerator),
    streamToHost_(streamToHost),
    streamAccelerator_(streamAccelerator),
    proc_(proc),
    mapDeviceAddresses_("device_addresses"),
    mapPinnedBuffers_("map_pinned_buffers"),
    changes_(false)
{
}

address_space::~address_space()
{
}

#if 0
core::io_buffer *
address_space::create_io_buffer(size_t count, GmacProtection prot)
{
    core::io_buffer *ret = new gmac::core::io_buffer(ctx_, count, prot);
    return ret;
}

gmacError_t
address_space::destroy_io_buffer(core::io_buffer &buffer)
{
    gmacError_t ret;
    ret = ctx_.free_buffer(buffer.get_buffer());

    delete &buffer;

    return ret;
}
#endif

gmacError_t
address_space::map(hal::ptr_t &dst, hostptr_t src, size_t count, unsigned align)
{
    gmacError_t ret;
    hal::ptr_t acc(0);
    if (align > 1) {
        count += align;
    }
#if 0
    bool hasMapping = getAccelerator().getMapping(acc, src, count);
    if (hasMapping == true) {
        ret = gmacSuccess;
        dst = acc;
        TRACE(LOCAL,"Mapping for address %p: %p", src, dst.get());
    } else {
        ret = getAccelerator().map(dst, src, count, align);
        TRACE(LOCAL,"New Mapping for address %p: %p", src, dst.get());
    }
#endif
    dst = ctx_.alloc(count, ret);

    return ret;
}

gmacError_t
address_space::unmap(hostptr_t addr, size_t count)
{
    gmacError_t ret = gmacSuccess;
    map_addresses::iterator it = mapDeviceAddresses_.find(addr);

    if (it != mapDeviceAddresses_.end()) {
        ret = ctx_.free(it->second);
        mapDeviceAddresses_.erase(it);
    } else {
        ret = gmacErrorInvalidValue;
    }
    return ret;
}

hal::ptr_t
address_space::alloc_host_pinned(size_t count, gmacError_t &err)
{
    hal::ptr_t ret;
    ret = ctx_.alloc_host_pinned(count, GMAC_PROT_READWRITE, err);

    // TODO: cache pinned allocations
    return ret;
}

gmacError_t
address_space::free_host_pinned(hal::ptr_t ptr)
{
    gmacError_t ret;

    ret = ctx_.free_host_pinned(ptr);

    return ret;
}

hal::ptr_t
address_space::get_host_pinned_mapping(hostptr_t ptr, gmacError_t &err)
{
    hal::ptr_t ret(0);
    map_buffers::iterator it;
    it = mapPinnedBuffers_.find(ptr);

    if (it != mapPinnedBuffers_.end()) {
        ret = it->second->get_device_addr();
        err = gmacSuccess;
    } else {
        err = gmacErrorInvalidValue;
    }

    return ret;
}

gmacError_t
address_space::copy(hal::ptr_t dst, const hal::ptr_t src, size_t count)
{
    gmacError_t ret;

    if (dst.is_device_ptr() &&
        src.is_device_ptr()) {
        // TODO: use the event
        // hal::event_t event =
        ctx_.copy(dst, src, count, streamAccelerator_, ret);
    } else if (dst.is_device_ptr() &&
               src.is_host_ptr()) {
        // TODO: use the event
        // hal::event_t event =
        ctx_.copy(dst, src, count, streamToAccelerator_, ret);
    } else if (dst.is_host_ptr() &&
               src.is_device_ptr()) {
        // TODO: use the event
        // hal::event_t event =
        ctx_.copy(dst, src, count, streamToHost_, ret);
    } else if (dst.is_host_ptr() &&
               src.is_host_ptr()) {
        // TODO: use the event
        // hal::event_t event =
        memcpy(dst.get_host_addr(),
               src.get_host_addr(), count);
    } else {
        FATAL("Unhandled case");
    }

    return ret;
}

#if 0
gmacError_t
address_space::copy(hostptr_t host, const hal::ptr_t acc, size_t count)
{
    gmacError_t ret;

    // TODO: use the event
    // hal::event_t event =
        ctx_.copy(host, acc, count, streamToHost_, ret);
    return ret;
}

gmacError_t
address_space::copy(hal::ptr_t dst, const hal::ptr_t src, size_t count)
{
    gmacError_t ret;

    // TODO: use the event
    // hal::event_t event =
        ctx_.copy(dst, src, count, streamAccelerator_, ret);
    return ret;
}
#endif

gmacError_t
address_space::copy(hal::ptr_t dst, hal::device_input &input, size_t count)
{
    gmacError_t ret;

    // TODO: use the event
    // hal::event_t event =
        ctx_.copy(dst, input, count, streamToAccelerator_, ret);
    return ret;
}

gmacError_t
address_space::copy(hal::device_output &output, const hal::ptr_t src, size_t count)
{
    gmacError_t ret;

    // TODO: use the event
    // hal::event_t event =
        ctx_.copy(output, src, count, streamToHost_, ret);
    return ret;
}

hal::event_t
address_space::copy_async(hal::ptr_t dst, const hal::ptr_t src, size_t count, gmacError_t &err)
{
    hal::event_t ret;

    if (dst.is_device_ptr() &&
        src.is_device_ptr()) {
        ret = ctx_.copy_async(dst, src, count, streamAccelerator_, err);
    } else if (dst.is_device_ptr() &&
               src.is_host_ptr()) {
        ret = ctx_.copy_async(dst, src, count, streamToAccelerator_, err);
    } else if (dst.is_host_ptr() &&
               src.is_device_ptr()) {
        ret = ctx_.copy_async(dst, src, count, streamToHost_, err);
    } else if (dst.is_host_ptr() &&
               src.is_host_ptr()) {
        memcpy(dst.get_host_addr(),
               src.get_host_addr(), count);
    } else {
        FATAL("Unhandled case");
    }

    return ret;

}

#if 0
hal::event_t
address_space::copy_async(hostptr_t host, const hal::ptr_t acc, size_t count, gmacError_t &err)
{
    // TODO: use the event
    hal::event_t ret = ctx_.copy_async(host, acc, count, streamToHost_, err);
    return ret;
}

hal::event_t
address_space::copy_async(hal::ptr_t dst, const hal::ptr_t src, size_t count, gmacError_t &err)
{
    // TODO: use the event
    hal::event_t ret = ctx_.copy_async(dst, src, count, streamAccelerator_, err);
    return ret;
}
#endif

hal::event_t
address_space::copy_async(hal::ptr_t dst, hal::device_input &input, size_t count, gmacError_t &err)
{
    hal::event_t ret = ctx_.copy_async(dst, input, count, streamToAccelerator_, err);
    return ret;
}

hal::event_t
address_space::copy_async(hal::device_output &output, const hal::ptr_t src, size_t count, gmacError_t &err)
{
    hal::event_t ret = ctx_.copy_async(output, src, count, streamToHost_, err);
    return ret;
}

gmacError_t
address_space::memset(hal::ptr_t addr, int c, size_t count)
{
    gmacError_t ret;

    // TODO: use the event
    // hal::event_t event =
        ctx_.memset(addr, c, count, streamAccelerator_, ret);
    return ret;
}

hal::event_t
address_space::memset_async(hal::ptr_t addr, int c, size_t count, gmacError_t &err)
{
    hal::event_t ret = ctx_.memset_async(addr, c, count, streamAccelerator_, err);

    return ret;
}

bool
address_space::is_integrated() const
{
    return ctx_.get_device().is_integrated();
}

kernel *
address_space::get_kernel(gmac_kernel_id_t k)
{
    kernel *ret = NULL;
    map_kernel::const_iterator i;
    i = kernels_.find(k);
    if (i != kernels_.end()) {
        ret = i->second;
    } else {
        const hal::kernel_t *kernelHal = ctx_.get_code_repository().get_kernel(k);
        if (kernelHal != NULL) {
            ret = new kernel(*kernelHal);
            kernels_.insert(map_kernel::value_type(k, ret));
        }
    }
    return ret;
}

#if 0
void
address_space::notify_pending_changes()
{
    changes_ = true;
}
#endif

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
