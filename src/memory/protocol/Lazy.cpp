#include "Lazy.h"

namespace gmac { namespace memory { namespace protocol {

gmacError_t Lazy::acquire(Object &obj)
{
    return gmacSuccess;
}

gmacError_t Lazy::release(Object &obj)
{
    return gmacSuccess;
}

gmacError_t Lazy::read(Object &obj, void *addr)
{
    return gmacSuccess;
}

gmacError_t Lazy::write(Object &obj, void *addr)
{
    return gmacSuccess;
}


} } }
