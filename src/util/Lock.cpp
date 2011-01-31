#include <string>

#include "Lock.h"

namespace __impl { namespace util {

__Lock::__Lock(const char *name) :
#if defined(USE_TRACE)
    exclusive_(false),
    name_(name)
#endif
{
#ifndef USE_TRACE
    UNREFERENCED_PARAMETER(name);
#endif
}
}}
