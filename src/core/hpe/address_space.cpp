#include "core/io_buffer.h"

#include "address_space.h"
#include "thread.h"
#include "vdevice.h"

namespace __impl { namespace core { namespace hpe {


context &
address_space::get_context()
{
    context *context = thread::get_context(*this);
    if (context == NULL) {
        context = proc_.get_resource_manager().create_context(util::GetThreadId(), *this);
        ASSERTION(context != NULL);
    }

    return *context;
}

address_space::address_space(hal::context_t &context,
                             hal::stream_t &streamLaunch,
                             hal::stream_t &streamToAccelerator,
                             hal::stream_t &streamToHost,
                             hal::stream_t &streamAccelerator,
                             process &proc) :
    gmac::util::mutex("address_space"),
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

gmacError_t
address_space::map(accptr_t &dst, hostptr_t src, size_t count, unsigned align)
{
    gmacError_t ret;
    accptr_t acc(0);
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

hostptr_t
address_space::alloc_host_pinned(size_t count, gmacError_t &err)
{
    hostptr_t ret(0);
    hal::buffer_t *buf = ctx_.alloc_buffer(count, GMAC_PROT_READWRITE, err);
    if (buf != NULL) {
        ret = buf->get_addr();

        mapPinnedBuffers_.insert(map_buffers::value_type(ret, buf));
    }

    // TODO: cache pinned allocations
    return ret;
}

gmacError_t
address_space::free_host_pinned(hostptr_t ptr)
{
    gmacError_t ret = gmacSuccess;
    map_buffers::iterator it;
    it = mapPinnedBuffers_.find(ptr);

    if (it != mapPinnedBuffers_.end()) {
        ctx_.free_buffer(*it->second);
        mapPinnedBuffers_.erase(it);
    } else {
        ret = gmacErrorInvalidValue;
    }

    return ret;
}

accptr_t
address_space::get_host_pinned_mapping(hostptr_t ptr, gmacError_t &err)
{
    accptr_t ret(0);
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
address_space::copy(accptr_t acc, const hostptr_t host, size_t count)
{
    context &context = get_context();
    gmacError_t ret;

    // TODO: use the event
    // hal::event_t *event =
        context.copy(acc, host, count, ret);
    return ret;
}

gmacError_t
address_space::copy(hostptr_t host, const accptr_t acc, size_t count)
{
    context &context = get_context();
    gmacError_t ret;

    // TODO: use the event
    // hal::event_t *event =
        context.copy(host, acc, count, ret);
    return ret;
}

gmacError_t
address_space::copy(accptr_t dst, const accptr_t src, size_t count)
{
    context &context = get_context();
    gmacError_t ret;

    // TODO: use the event
    // hal::event_t *event =
        context.copy(dst, src, count, ret);
    return ret;
}

hal::async_event_t *
address_space::copy_async(accptr_t acc, const hostptr_t host, size_t count, gmacError_t &err)
{
    context &context = get_context();

    // TODO: use the event
    hal::async_event_t *ret = context.copy_async(acc, host, count, err);
    return ret;
}

hal::async_event_t *
address_space::copy_async(hostptr_t host, const accptr_t acc, size_t count, gmacError_t &err)
{
    context &context = get_context();

    // TODO: use the event
    hal::async_event_t *ret = context.copy_async(host, acc, count, err);
    return ret;
}

hal::async_event_t *
address_space::copy_async(accptr_t dst, const accptr_t src, size_t count, gmacError_t &err)
{
    context &context = get_context();

    // TODO: use the event
    hal::async_event_t *ret = context.copy_async(dst, src, count, err);
    return ret;
}

gmacError_t
address_space::copy(accptr_t dst, core::io_buffer &buffer, size_t off, size_t count)
{
    context &context = get_context();
    gmacError_t ret;

    if (buffer.async()) {
        ret = context.copy_async(dst, buffer, off, count);
    } else {
        ret = context.copy(dst, buffer, off, count);
    }
    return ret;
}

gmacError_t
address_space::copy(core::io_buffer &buffer, size_t off, const accptr_t src, size_t count)
{
    context &context = get_context();
    gmacError_t ret;

    if (buffer.async()) {
        ret = context.copy_async(buffer, off, src, count);
    } else {
        ret = context.copy(buffer, off, src, count);
    }
    return ret;
}

gmacError_t
address_space::memset(accptr_t addr, int c, size_t count)
{
    context &context = get_context();
    gmacError_t ret;

    // TODO: use the event
    // hal::event_t *event =
        context.memset(addr, c, count, ret);
    return ret;
}

hal::async_event_t *
address_space::memset_async(accptr_t addr, int c, size_t count, gmacError_t &err)
{
    context &context = get_context();

    hal::async_event_t *ret = context.memset_async(addr, c, count, err);

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
    kernel *ker = NULL;
    map_kernel::const_iterator i;
    i = kernels_.find(k);
    if (i != kernels_.end()) {
        ker = i->second;
    }
    return ker;
}

void
address_space::register_kernel(gmac_kernel_id_t k, const hal::kernel_t &ker)
{
    TRACE(LOCAL,"CTX: %p Registering kernel %s: %p", this, ker.get_name().c_str(), k);
    map_kernel::iterator i;
    i = kernels_.find(k);
    ASSERTION(i == kernels_.end());
    kernel *kernelNew = new kernel(ker);
    kernels_[k] = kernelNew;
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
