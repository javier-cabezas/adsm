#include "CoherenceDomain.h"
#include "Device.h"

namespace __impl { namespace hal { namespace cuda {

CoherenceDomain::CoherenceDomain() :
    hal::detail::CoherenceDomain<Device>()
{
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
