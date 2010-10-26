#include "Kernel.h"

#include <cstring>
#include <algorithm>

namespace gmac {

KernelConfig::KernelConfig(const KernelConfig & c) :
    argsSize_(0)
{
    ArgVector::const_iterator it;
    for (it = c.begin(); it != c.end(); it++) {
        pushArgument(it->ptr_, it->size_, it->offset_);
    }
}

void KernelConfig::pushArgument(const void *arg, size_t size, off_t offset)
{
#if 0
	if (size == 4) {
        trace("Pushing argument: +%zd, %" PRId64 "/%" PRId64 ": 0x%x", size, argsSize_, offset, *(uint32_t *) arg);
    } else if (size == 8) {
        trace("Pushing argument: +%zd, %" PRId64 "/%" PRId64 ": %p", size, argsSize_, offset, (void *) *(uint64_t *) arg);
    } else {
        trace("Pushing argument: +%zd, %" PRId64 "/%" PRId64, size, argsSize_, offset);
    }
#endif

    assertion(offset + size < KernelConfig::StackSize_);

    memcpy(&stack_[offset], arg, size);
    argsSize_ = (off_t)(offset + size);
    push_back(Argument(&stack_[offset], size, offset));
}


}

