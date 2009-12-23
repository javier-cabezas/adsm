#ifndef __ACCELERATOR_IPP_
#define __ACCELERATOR_IPP_

inline CUdevice
Accelerator::device() const
{
    return _device;
}

inline size_t
Accelerator::memory() const
{
    return _memory;
}

inline bool
Accelerator::async() const
{
    return _async;
}

#endif
