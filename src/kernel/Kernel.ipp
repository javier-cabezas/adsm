#ifndef __KERNEL_KERNEL_IPP
#define __KERNEL_KERNEL_IPP

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <kernel/Mode.h>
#include <memory/Object.h>

#include <cstring>
#include <algorithm>

namespace gmac {

inline
Kernel::Kernel(const KernelDescriptor & k) :
    KernelDescriptor(k.name(), k.key())
{
}

#if 0
inline
gmacError_t Kernel::bind(void * addr)
{
    gmacError_t ret;
    ret = gmacErrorInvalidValue;
    memory::Object * object =
        Mode::current()->map().find(addr);

    if (object != NULL) {
        const_iterator j;

        j = std::find(begin(), end(), object);

        if (j == end()) {
            ret = gmacSuccess;
            insert(object);
        } else {
            ret = gmacErrorAlreadyBound;
        }
    }

    return ret;
}

inline
gmacError_t Kernel::unbind(void * addr)
{
    gmacError_t ret;
    ret = gmacErrorInvalidValue;
    memory::Object * object =
        Mode::current()->map().find(addr);

    if (object != NULL) {
        iterator j;

        j = std::find(begin(), end(), object);

        if (j != end()) {
            ret = gmacSuccess;
            erase(j);
        }
        return ret;
    }

    return ret;
}
#endif

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

inline
off_t KernelConfig::argsSize() const
{
    return _argsSize;
}

inline
char * KernelConfig::argsArray()
{
    return _stack;
}

}

#endif
