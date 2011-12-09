#include "core/hpe/address_space.h"
#include "core/hpe/kernel.h"
#include "core/hpe/process.h"
#include "core/hpe/vdevice.h"

#include "hal/types.h"

namespace __impl { namespace core { namespace hpe {

vdevice::vdevice(process &proc, util::shared_ptr<address_space> aspace,
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

kernel::launch_ptr
vdevice::launch(gmac_kernel_id_t id, hal::kernel_t::config &conf,
                                     hal::kernel_t::arg_list &args, gmacError_t &err)
{
    kernel::launch_ptr ret;
    kernel *k = get_address_space()->get_kernel(id);
    if (k != NULL) {
        ret = k->launch_config(*this, conf, args, streamLaunch_, err);
    } else {
        err = gmacErrorInvalidValue;
    }
    return ret;
}

hal::event_t
vdevice::execute(kernel::launch_ptr launch, gmacError_t &err)
{
    hal::event_t ret;
    err = gmacSuccess;

    if (launch->get_arg_list().get_objects().size() == 0) {
        hal::event_t event = aspace_->streamToAccelerator_.get_last_event();
        ret = launch->execute(event, err);
    } else {
        // TODO: Implement per object synchronization
        hal::event_t event = aspace_->streamToAccelerator_.get_last_event();
        ret = launch->execute(event, err);
    }

    return ret;
}

gmacError_t
vdevice::wait(kernel::launch_ptr launch)
{
    gmacError_t ret = gmacSuccess;
    hal::event_t event = launch->get_event();

    ret = event.sync();

    return ret;
}

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
