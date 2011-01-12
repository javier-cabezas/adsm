#include <cstring>
#include <algorithm>

#include "util/Logger.h"

#include "Kernel.h"

namespace __impl { namespace core {

KernelConfig::KernelConfig(const KernelConfig & c) :
    argsSize_(0)
{
    ArgVector::const_iterator it;
    for (it = c.begin(); it != c.end(); it++) {
        pushArgument(it->ptr(), it->size(), it->offset());
    }
}

void KernelConfig::pushArgument(const void *arg, size_t size, unsigned long offset)
{
    ASSERTION(offset + size < KernelConfig::StackSize_);

    memcpy(&stack_[offset], arg, size);
    push_back(Argument(&stack_[offset], size, offset));
    argsSize_ = size_t(offset) + size;
}

void KernelConfig::pushArgument(const void *arg, size_t size)
{
    pushArgument(arg, size, argsSize_);
}

}}
