#ifndef __GPU_IPP_
#define __GPU_IPP_

inline CUdevice
GPU::device() const
{
    return _device;
}

inline size_t
GPU::memory() const
{
    return _memory;
}

inline bool
GPU::async() const
{
    return _async;
}

#endif
