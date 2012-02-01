#ifndef GMAC_CORE_HPE_VIRTUAL_DEVICE_TABLE_IMPL_H_
#define GMAC_CORE_HPE_VIRTUAL_DEVICE_TABLE_IMPL_H_

namespace __impl { namespace core { namespace hpe {

inline
vdevice_table::vdevice_table() :
    Parent()
{
}

inline
gmacError_t
vdevice_table::add_virtual_device(GmacVirtualDeviceId id, vdevice &vDevice)
{
    Parent::iterator it = Parent::find(id);
    ASSERTION(it == Parent::end(), "Virtual device not found");

    Parent::insert(Parent::value_type(id, &vDevice));

    return gmacSuccess;
}

inline
gmacError_t
vdevice_table::remove_virtual_device(GmacVirtualDeviceId id)
{
    Parent::iterator it = Parent::find(id);
    ASSERTION(it != Parent::end(), "Virtual device not found");

    Parent::erase(it);

    return gmacSuccess;
}

inline
vdevice *
vdevice_table::get_virtual_device(GmacVirtualDeviceId id)
{
    vdevice *ret = NULL;

    Parent::iterator it = Parent::find(id);

    if (it != Parent::end()) {
        ret = it->second;
    }

    return ret;
}

}}}

#endif // GMAC_CORE_HPE_VIRTUAL_DEVICE_TABLE_IMPL_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
