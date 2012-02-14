#ifndef GMAC_HAL_PLATFORM_IMPL_H_
#define GMAC_HAL_PLATFORM_IMPL_H_

namespace __impl { namespace hal { namespace detail {

inline
platform::~platform()
{
    for (list_device::iterator it  = devices_.begin();
                               it != devices_.end();
                             ++it) {
        delete (*it);
    }

    devices_.clear();
}

inline
void
platform::add_device(device &d)
{
    devices_.push_back(&d);
}

inline
unsigned
platform::get_ndevices()
{
    return unsigned(devices_.size());
}

inline
platform::list_device
platform::get_devices()
{
    return devices_;
}

inline
platform::list_device
platform::get_devices(device::type type)
{
    list_device ret;

    for (list_device::iterator it  = devices_.begin();
                               it != devices_.end();
                               it++) {
        if ((*it)->get_type() == type) {
            ret.push_back(*it);
        }
    }
    return ret;
}

}}}

#endif


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
