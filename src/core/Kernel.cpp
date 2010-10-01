#include "Kernel.h"

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <cstring>
#include <algorithm>

namespace gmac {

KernelConfig::KernelConfig(const KernelConfig & c) :
    argsSize_(0)
{
    ArgVector::const_iterator it;
    for (it = c.begin(); it != c.end(); it++) {
        pushArgument(it->_ptr, it->_size, it->_offset);
    }
}

void KernelConfig::pushArgument(const void *arg, size_t size, off_t offset)
{
    if (size == 4) {
        trace("Pushing argument: +%zd, %" PRId64 "/%" PRId64 ": 0x%x", size, argsSize_, offset, *(uint32_t *) arg);
    } else if (size == 8) {
        trace("Pushing argument: +%zd, %" PRId64 "/%" PRId64 ": %p", size, argsSize_, offset, (void *) *(uint64_t *) arg);
    } else {
        trace("Pushing argument: +%zd, %" PRId64 "/%" PRId64, size, argsSize_, offset);
    }

    assertion(offset + size < KernelConfig::StackSize);

    memcpy(&stack_[offset], arg, size);
    argsSize_ = offset + size;
    push_back(Argument(&stack_[offset], size, offset));
}


}

