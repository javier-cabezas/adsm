#ifndef __KERNEL_IOBUFFER_IPP_
#define __KERNEL_IOBUFFER_IPP_

#include "Mode.h"

namespace gmac {

inline
IOBuffer::IOBuffer(size_t __size) :
    util::Lock(paraver::LockIo),
    __addr(NULL),
    __size(__size)
{ }



}


#endif
