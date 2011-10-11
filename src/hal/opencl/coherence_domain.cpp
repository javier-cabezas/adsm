#include "coherence_domain.h"
#include "device.h"

namespace __impl { namespace hal { namespace opencl {

coherence_domain::coherence_domain() :
    hal::detail::coherence_domain<device>()
{
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
