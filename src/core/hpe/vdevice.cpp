#include "memory/Manager.h"
#include "memory/object.h"

#include "core/io_buffer.h"

#include "core/hpe/address_space.h"
#include "core/hpe/kernel.h"
#include "core/hpe/vdevice.h"
#include "core/hpe/context.h"
#include "core/hpe/process.h"

#include "hal/types.h"

namespace __impl { namespace core { namespace hpe {

vdevice::vdevice(process &proc, util::smart_ptr<address_space>::shared aspace,
                 hal::stream_t &streamLaunch) :
    proc_(proc),
    aspace_(aspace),
    streamLaunch_(streamLaunch)
#if defined(_MSC_VER)
#	pragma warning( disable : 4355 )
#endif
#ifdef USE_VM
    , bitmap_(*this)
#endif
{
}

vdevice::~vdevice()
{
}

#if 0
std::string
vdevice::getKernelName(gmac_kernel_id_t k) const
{
    KernelMap::const_iterator i;
    i = kernels_.find(k);
    ASSERTION(i != kernels_.end());
    return i->second->get_name();
}
#endif

kernel::launch *
vdevice::launch(gmac_kernel_id_t id, hal::kernel_t::config &config, gmacError_t &err)
{
#if 0
    kernel::launch *ret = NULL;
    KernelMap::const_iterator i;
    i = kernels_.find(id);
    if (i != kernels_.end()) {
        kernel *k = i->second;
        ret = k->launch_config(config, *this, *streamLaunch_, err);
    } else {
        err = gmacErrorInvalidValue;
    }
    return ret;
#endif
    kernel::launch *ret;
    kernel *k = get_address_space()->get_kernel(id);
    if (k != NULL) {
        ret = k->launch_config(config, *this, streamLaunch_, err);
    } else {
        err = gmacErrorInvalidValue;
    }
    return ret;
}

hal::async_event_t *
vdevice::execute(kernel::launch &launch, gmacError_t &err)
{
    hal::async_event_t *ret;
    err = gmacSuccess;

    if (launch.get_objects().size() == 0) {
        hal::async_event_t *event = aspace_->streamToAccelerator_.get_last_async_event();
        if (event != NULL) {
            ret = launch.execute(*event, err);
        } else {
            ret = launch.execute(err);
        }
    } else {
        // TODO: Implement per object synchronization
        hal::async_event_t *event = aspace_->streamToAccelerator_.get_last_async_event();
        if (event != NULL) {
            ret = launch.execute(*event, err);
        } else {
            ret = launch.execute(err);
        }
    }

    return ret;
}

gmacError_t
vdevice::wait(kernel::launch &launch)
{
    gmacError_t ret = gmacSuccess;
    hal::async_event_t *event = launch.get_event();
    if (event != NULL) {
        ret = event->sync();
    }

    return ret;
}

#if 0
gmacError_t
vdevice::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t count)
{
    TRACE(LOCAL,"Copy %p to accelerator %p ("FMT_SIZE" bytes)", host, acc.get(), count);

    switchIn();
    gmacError_t ret = getContext().copyToAccelerator(acc, host, count);
    switchOut();

    return ret;
}

gmacError_t
vdevice::copyToHost(hostptr_t host, const accptr_t acc, size_t count)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", acc.get(), host, count);

    switchIn();
    gmacError_t ret = getContext().copyToHost(host, acc, count);
    switchOut();

    return ret;
}

gmacError_t
vdevice::copyAccelerator(accptr_t dst, const accptr_t src, size_t count)
{
    switchIn();
    gmacError_t ret = getAccelerator().copyAccelerator(dst, src, count, *streamToHost_);
    switchOut();
    return ret;
}

gmacError_t
vdevice::memset(accptr_t addr, int c, size_t count)
{
    switchIn();
    gmacError_t ret = getAccelerator().memset(addr, c, count, *streamLaunch_);
    switchOut();
    return ret;
}

// Nobody can enter GMAC until this has finished. No locks are needed
gmacError_t
vdevice::moveTo(Accelerator &acc)
{
    TRACE(LOCAL,"Moving mode from acc %d to %d", getAccelerator().id(), acc.id());
    switchIn();

    if (acc_ == &acc) {
        switchOut();
        return gmacSuccess;
    }
    gmacError_t ret = gmacSuccess;
    size_t free;
    size_t total;
    size_t needed = context_.memorySize();
    getAccelerator().getMemInfo(free, total);

    if (needed > free) {
        switchOut();
        return gmacErrorInsufficientAcceleratorMemory;
    }

    TRACE(LOCAL,"Releasing object memory in accelerator");
    ret = context_.forEachObject(&memory::object::unmapFromAccelerator);

    TRACE(LOCAL,"Cleaning contexts");
    contextMap_.clean();

    TRACE(LOCAL,"Registering mode in new accelerator");
    getAccelerator().migratevdevice(*this, acc);

    TRACE(LOCAL,"Reallocating objects");
    // context_.reallocObjects(*this);
    ret = context_.forEachObject(&memory::object::mapToAccelerator);

    TRACE(LOCAL,"Reloading mode");
    reload();

    switchOut();

    return ret;
}
#endif

void
vdevice::getMemInfo(size_t &free, size_t &total)
{
    free = aspace_->get_hal_context().get_device().get_free_memory();
    total = aspace_->get_hal_context().get_device().get_total_memory();
}

gmacError_t
vdevice::cleanUp()
{
#ifdef USE_VM
    bitmap_.cleanUp();
#endif
    return gmacSuccess;
}

}}}
