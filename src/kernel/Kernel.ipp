#ifndef __KERNEL_KERNEL_IPP
#define __KERNEL_KERNEL_IPP

#include "Context.h"

#include <algorithm>

namespace gmac {

inline
Kernel::Kernel(const KernelDescriptor & k) :
    KernelDescriptor(k.name(), k.key())
{
}

inline
gmacError_t
Kernel::bind(void * addr)
{
    gmacError_t ret;
    ret = gmacErrorInvalidValue;
    Context * ctx = Context::current();
    memory::Region * region = ctx->mm().find<memory::Region>(addr);

    if (region != NULL) {
        const_iterator j;

        j = std::find(begin(), end(), region);

        if (j == end()) {
            ret = gmacSuccess;
            push_back(region);
        } else {
            ret = gmacErrorAlreadyBound;
        }
    }

    return ret;
}

inline
gmacError_t
Kernel::unbind(void * addr)
{
    gmacError_t ret;
    ret = gmacErrorInvalidValue;
    Context * ctx = Context::current();
    memory::Region * region = ctx->mm().find<memory::Region>(addr);

    if (region != NULL) {
        iterator j;

        j = std::find(begin(), end(), region);

        if (j != end()) {
            ret = gmacSuccess;
            erase(j);
        }
        return ret;
    }

    return ret;
}

inline
Argument::Argument(void * ptr, size_t size, off_t offset) :
    _ptr(ptr),
    _size(size),
    _offset(offset)
{}

inline
KernelConfig::KernelConfig(const KernelConfig & c) :
    _argsSize(0)
{
    ArgVector::const_iterator it;
    for (it = c.begin(); it != c.end(); it++) {
        pushArgument(it->_ptr, it->_size, it->_offset);
    }
}

inline
KernelConfig::KernelConfig() :
    ArgVector(),
    _argsSize(0)
{
}

inline
KernelConfig::~KernelConfig()
{
}

inline
void
KernelConfig::pushArgument(const void *arg, size_t size, off_t offset)
{
    TRACE("Pushing argument: +%d, %d/%d: %p", size, _argsSize, offset, (void *) *(long int *) arg);
    assert(offset + size < KernelConfig::StackSize);

    memcpy(&_stack[offset], arg, size);
    _argsSize = offset + size;
    push_back(Argument(&_stack[offset], size, offset));
}

inline
off_t
KernelConfig::argsSize() const
{
    return _argsSize;
}

inline
char *
KernelConfig::argsArray()
{
    return _stack;
}

}

#endif
