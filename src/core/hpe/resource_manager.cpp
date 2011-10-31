#include <algorithm>

#include "core/hpe/resource_manager.h"

#include "core/hpe/address_space.h"
#include "core/hpe/context.h"
#include "core/hpe/vdevice.h"
#include "core/hpe/vdevice_table.h"
#include "core/hpe/thread.h"

namespace __impl { namespace core { namespace hpe {

#if 0
inline
ContextMap::ContextMap(address_space &owner) :
    gmac::util::RWLock("ContextMap"), owner_(owner)
{
}

inline void
ContextMap::add(THREAD_T id, context &ctx)
{
    lockWrite();
    Parent::insert(Parent::value_type(id, &ctx));
    unlock();
}

inline context *
ContextMap::find(THREAD_T id)
{
    lockRead();
    Parent::iterator i = Parent::find(id);
    context *ret = NULL;
    if(i != end()) ret = i->second;
    unlock();
    return ret;
}

inline void
ContextMap::remove(THREAD_T id)
{
    lockWrite();
    Parent::erase(id);
    unlock();
}

inline void
ContextMap::clean()
{
    Parent::iterator i;
    lockWrite();
    for(i = begin(); i != end(); i++) {
        delete i->second;
    }
    Parent::clear();
    unlock();
}

#endif

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

#if 0
    address_space *aspace = new address_space(*resources.context_,
                                              *resources.streamLaunch_,
                                              *resources.streamToAccelerator_,
                                              *resources.streamToHost_,
                                              *resources.streamAccelerator_,
                                              proc_);

    context *context = get_context();
    if (context == NULL) {
        context = new gmac::core::hpe::context(ctx_, *streamLaunch,
                                                     *streamToAccelerator,
                                                     *streamToHost_,
                                                     *streamAccelerator);
        contextMap_.add(util::GetThreadId(), *context);
    }
    return *context;
#endif

    return ctx;
}

resource_manager::resource_manager(process &proc) :
    proc_(proc)
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
        address_space *aspace = create_address_space(0, ret);
        if (ret == gmacSuccess) {
            dev = create_virtual_device(*aspace, ret);
        }
    } else {
        dev = create_virtual_device(parent->get_current_virtual_device().get_address_space(), ret);
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

address_space *
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
#if 0
    resources.streamToAccelerator_ = device.create_stream(*resources.context_);
    resources.streamToHost_        = device.create_stream(*resources.context_);
    resources.streamAccelerator_   = device.create_stream(*resources.context_);
#endif

    address_space *aspace = new address_space(*resources.context_,
                                              *resources.streamLaunch_,
                                              *resources.streamToAccelerator_,
                                              *resources.streamToHost_,
                                              *resources.streamAccelerator_,
                                              proc_);
    ASSERTION(aspace != NULL);
    aspaceMap_.insert(map_aspace::value_type(aspace->get_id(), aspace));
    aspaceResourcesMap_.insert(map_aspace_resources::value_type(aspace, resources));

    resources.context_->get_code_repository().register_kernels(*aspace);

    return aspace;
}

gmacError_t
resource_manager::destroy_address_space(address_space &aspace)
{
    gmacError_t ret = gmacSuccess;

    map_aspace::iterator it;
    it = aspaceMap_.find(aspace.get_id());

    if (it != aspaceMap_.end()) {
        aspaceMap_.erase(it);
    } else {
        ret = gmacErrorInvalidValue;
    }

    return ret;
}

vdevice *
resource_manager::create_virtual_device(address_space &aspace, gmacError_t &err)
{
    vdevice *ret = NULL;
    err = gmacSuccess;

    map_aspace::iterator it;
    it = aspaceMap_.find(aspace.get_id());

    if (it != aspaceMap_.end()) {
        address_space &aspace = *it->second;

        TRACE(LOCAL,"Creatintg Execution vdevice on aspace#"FMT_ASPACE, aspace.get_id().val);

        // Initialize the global shared memory for the context
        ret = new vdevice(proc_, aspace, aspace.streamLaunch_);
    } else {
        err = gmacErrorInvalidValue;
    }

    return ret;
}

gmacError_t
resource_manager::destroy_virtual_device(vdevice &dev)
{
    gmacError_t ret = gmacSuccess;

    ret = thread::remove_virtual_device(dev);

    return ret;
}

#if 0
vdevice *
resource_manager::getVirtualDevice(GmacVirtualDeviceId vDeviceId)
{
    vdevice *ret = NULL;

    vdevice_table &vDeviceTable = thread::getCurrentVirtualDeviceTable();
    ret = vDeviceTable.getVirtualDevice(vDeviceId);

    return ret;
}
#endif

address_space *
resource_manager::get_address_space(GmacAddressSpaceId aSpaceId)
{
    address_space *ret = NULL;

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
