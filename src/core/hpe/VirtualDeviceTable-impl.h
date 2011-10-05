#ifndef GMAC_CORE_HPE_VIRTUAL_DEVICE_TABLE_IMPL_H_
#define GMAC_CORE_HPE_VIRTUAL_DEVICE_TABLE_IMPL_H_

namespace __impl { namespace core { namespace hpe {

inline
VirtualDeviceTable::VirtualDeviceTable() :
    Parent()
{
}

inline
gmacError_t
VirtualDeviceTable::addVirtualDevice(GmacVirtualDeviceId vDeviceId, Mode &vDevice)
{
    Parent::iterator it = Parent::find(vDeviceId);
    ASSERTION(it == Parent::end(), "Virtual device not found");

    Parent::insert(Parent::value_type(vDeviceId, &vDevice));

    return gmacSuccess;
}

inline
gmacError_t
VirtualDeviceTable::removeVirtualDevice(GmacVirtualDeviceId vDeviceId)
{
    Parent::iterator it = Parent::find(vDeviceId);
    ASSERTION(it != Parent::end(), "Virtual device not found");

    Parent::erase(it);

    return gmacSuccess;
}

inline
Mode *
VirtualDeviceTable::getVirtualDevice(GmacVirtualDeviceId vDeviceId)
{
    Mode *ret = NULL;

    Parent::iterator it = Parent::find(vDeviceId);

    if (it != Parent::end()) {
        ret = it->second;
    }

    return ret;
}

}}}

#endif // GMAC_CORE_HPE_VIRTUAL_DEVICE_TABLE_IMPL_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
