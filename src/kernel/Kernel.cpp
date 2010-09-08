#include "Kernel.h"

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <cstring>
#include <algorithm>

namespace gmac {

KernelConfig::KernelConfig(const KernelConfig & c) :
    _argsSize(0)
{
    ArgVector::const_iterator it;
    for (it = c.begin(); it != c.end(); it++) {
        pushArgument(it->_ptr, it->_size, it->_offset);
    }
}

void KernelConfig::pushArgument(const void *arg, size_t size, off_t offset)
{
    if (size == 4) {
        trace("Pushing argument: +%zd, %" PRId64 "/%" PRId64 ": 0x%x", size, _argsSize, offset, *(uint32_t *) arg);
    } else if (size == 8) {
        trace("Pushing argument: +%zd, %" PRId64 "/%" PRId64 ": %p", size, _argsSize, offset, (void *) *(uint64_t *) arg);
    } else {
        trace("Pushing argument: +%zd, %" PRId64 "/%" PRId64, size, _argsSize, offset);
    }

    assertion(offset + size < KernelConfig::StackSize);

    memcpy(&_stack[offset], arg, size);
    _argsSize = offset + size;
    push_back(Argument(&_stack[offset], size, offset));
}


}

