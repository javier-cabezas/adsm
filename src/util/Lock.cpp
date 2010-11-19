#include <string>

#include "Lock.h"

namespace __impl { namespace util {

__Lock::__Lock(const char *name) 
{
    UNREFERENCED_PARAMETER(name);

#if defined(USE_TRACE)
    exclusive_ = false;
#endif
}
}}
