#include "platform.h"

namespace __impl { namespace hal { namespace detail {

platform::list_device
platform::get_devices(device::type type)
{
    list_device ret;

    for (list_device::iterator it  = devices_.begin();
                               it != devices_.end();
                             ++it) {
        if ((*it)->get_type() == type) {
            ret.push_back(*it);
        }
    }
    return devices_;
}


}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
