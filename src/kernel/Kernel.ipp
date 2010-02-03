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
    memory::Region * region = ctx->mm().localFind(addr);

    if (region != NULL) {
        RegionVector::const_iterator j;

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
    memory::Region * region = ctx->mm().localFind(addr);

    if (region != NULL) {
        RegionVector::iterator j;

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
Argument::Argument(void * ptr, size_t size) :
    _ptr(ptr),
    _size(size)
{}

inline
KernelConfig::KernelConfig(const KernelConfig & c) :
    _argsSize(0)
{
    ArgVector::const_iterator it;
    for (it = c.begin(); it != c.end(); it++) {
        pushArgument(it->_ptr, it->_size, _argsSize);
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
    assert(_argsSize == offset);
    assert(_argsSize + size < KernelConfig::StackSize);

    memcpy(&_stack[offset], arg, size);
    _argsSize += size;
    push_back(Argument(&_stack[offset], size));
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
