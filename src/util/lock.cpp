#include <string>

#include "lock.h"

namespace __impl { namespace util {

lock__::lock__(const char *name)
#if defined(USE_TRACE_LOCKS)
    : exclusive_(false),
    name_(name)
#endif
{
#ifndef USE_TRACE_LOCKS
    UNREFERENCED_PARAMETER(name);
#endif
}
}}
