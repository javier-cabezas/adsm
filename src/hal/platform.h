#ifndef GMAC_HAL_PLATFORM_H_
#define GMAC_HAL_PLATFORM_H_

#include <set>

#include "config/common.h"

namespace __impl { namespace hal {

namespace detail {

class coherence_domain;

class device;

class GMAC_LOCAL platform :
    public util::unique<platform>
{
public:
    typedef std::list<device *> list_device;
private:
    list_device devices_;
public:
    virtual ~platform();
    void add_device(device &d);
    unsigned get_ndevices();

    list_device get_devices();
    list_device get_devices(device::type type);
};

}

}}

#include "platform-impl.h"

#endif /* GMAC_HAL_DEVICE_H_ */


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
