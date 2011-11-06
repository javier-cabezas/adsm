#include <algorithm>

#include "core/hpe/resource_manager.h"

#include "core/hpe/address_space.h"
// #include "core/hpe/context.h"
#include "core/hpe/vdevice.h"
#include "core/hpe/vdevice_table.h"
#include "core/hpe/thread.h"

namespace __impl { namespace core { namespace hpe {

#if 0
context *
resource_manager::create_context(THREAD_T id, address_space &aspace)
{
    context *ctx = NULL;

    map_aspace_resources::iterator it = aspaceResourcesMap_.find(&aspace);

    if (it != aspaceResourcesMap_.end()) {
        address_space_resources &resources = it->second;
        ctx = new context(*resources.context_, *resources.streamLaunch_,
                                               *resources.streamToAccelerator_,
                                               *resources.streamToHost_,
                                               *resources.streamAccelerator_);
    }

    return ctx;
}
#endif

resource_manager::resource_manager(process &proc) :
    proc_(proc),
    aspaceMap_("aspace_map"),
    aspaceResourcesMap_("aspace_resources_map")
{
}

resource_manager::~resource_manager()
{
    std::vector<hal::device *>::iterator it;
    for (it = devices_.begin(); it != devices_.end(); ++it) {
        hal::device *dev = *it;
        ASSERTION(dev != NULL);
        delete dev;
    }
}

gmacError_t
resource_manager::init_thread(thread &t, const thread *parent)
{
    gmacError_t ret;
    vdevice *dev = NULL;

    if (parent == NULL) {
        address_space_ptr aspace = create_address_space(0, ret);
        if (ret == gmacSuccess) {
            dev = create_virtual_device(aspace->get_id(), ret);
        }
    } else {
        dev = create_virtual_device(parent->get_current_virtual_device().get_address_space()->get_id(), ret);
    }

    if (ret == gmacSuccess) {
        ret = t.add_virtual_device(dev->get_id(), *dev);
        t.set_current_virtual_device(*dev);
    }

    return ret;
}

gmacError_t
resource_manager::register_device(hal::device &dev)
{
    ASSERTION(std::find(devices_.begin(), devices_.end(), &dev) == devices_.end(),
              "Device already registered");

    devices_.push_back(&dev);

    return gmacSuccess;
}

address_space_ptr
resource_manager::create_address_space(unsigned accId, gmacError_t &err)
{
    err = gmacSuccess;
    hal::device &device = *devices_[accId];

    // Lets use a different context per address space for now
    address_space_resources resources;
    resources.context_ = device.create_context();

    resources.streamLaunch_        = device.create_stream(*resources.context_);
    resources.streamToAccelerator_ = device.create_stream(*resources.context_);
    resources.streamToHost_        = device.create_stream(*resources.context_);
    resources.streamAccelerator_   = device.create_stream(*resources.context_);

    address_space *aspace = new address_space(*resources.context_,
                                              *resources.streamLaunch_,
                                              *resources.streamToAccelerator_,
                                              *resources.streamToHost_,
                                              *resources.streamAccelerator_,
                                              proc_);
    ASSERTION(aspace != NULL);
    address_space_ptr ptrAspace(aspace);
    aspaceMap_.insert(map_aspace::value_type(aspace->get_id(), ptrAspace));
    aspaceResourcesMap_.insert(map_aspace_resources::value_type(aspace, resources));

    resources.context_->get_code_repository().register_kernels(*aspace);

    return ptrAspace;
}

gmacError_t
resource_manager::destroy_address_space(address_space &aspace)
{
    gmacError_t ret = gmacSuccess;

    map_aspace::iterator it;
    it = aspaceMap_.find(aspace.get_id());

    if (it != aspaceMap_.end()) {
        map_aspace_resources::size_type size = aspaceResourcesMap_.erase(it->second.get());
        ASSERTION(size == 1, "Resources not found for address space");
        aspaceMap_.erase(it);
    } else {
        ret = gmacErrorInvalidValue;
    }

    return ret;
}

vdevice *
resource_manager::create_virtual_device(GmacAddressSpaceId id, gmacError_t &err)
{
    vdevice *ret = NULL;
    err = gmacSuccess;

    map_aspace::iterator it;
    it = aspaceMap_.find(id);

    if (it != aspaceMap_.end()) {
        address_space_ptr aspace = it->second;

        TRACE(LOCAL,"Creatintg Execution vdevice on aspace#"FMT_ASPACE, aspace->get_id().val);

        // Initialize the global shared memory for the context
        ret = new vdevice(proc_, aspace, aspace->streamLaunch_);
    } else {
        err = gmacErrorInvalidValue;
    }

    return ret;
}

gmacError_t
resource_manager::destroy_virtual_device(vdevice &dev)
{
    gmacError_t ret = gmacSuccess;

    ret = thread::get_current_thread().remove_virtual_device(dev);

    return ret;
}

address_space_ptr
resource_manager::get_address_space(GmacAddressSpaceId aSpaceId)
{
    address_space_ptr ret;

    map_aspace::iterator it;
    it = aspaceMap_.find(aSpaceId);

    if (it != aspaceMap_.end()) {
        ret = it->second;
    }

    return ret;
}

unsigned
resource_manager::get_number_of_devices() const
{
    return unsigned(devices_.size());
}

bool
resource_manager::are_all_devices_integrated() const
{
    bool ret = true;
    std::vector<hal::device *>::const_iterator dev;
    for(dev = devices_.begin(); dev != devices_.end(); dev++) {
        ret = ret && (*dev)->is_integrated();
    }
    return ret;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
