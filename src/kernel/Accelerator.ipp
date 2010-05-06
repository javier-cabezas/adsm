#ifndef __KERNEL_ACCELERATOR_IPP_
#define __KERNEL_ACCELERATOR_IPP_

namespace gmac {

inline
size_t
Accelerator::memory() const
{
    return _memory;
}

inline unsigned
Accelerator::id() const
{
    return _id;
}

}

#endif
