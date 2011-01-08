#ifndef GMAC_CONFIG_CUDA_COMMON_H_
#define GMAC_CONFIG_CUDA_COMMON_H_

#include <cuda.h>

struct accptr_t {
    CUdeviceptr ptr_;

    inline accptr_t() :
        ptr_(NULL)
    {}
    inline accptr_t(void * ptr) :
        ptr_(CUdeviceptr((unsigned long )ptr & ((0xff << sizeof(CUdeviceptr)) - 1)))
    {}
    inline accptr_t(CUdeviceptr ptr) :
        ptr_(ptr)
    {}
    inline accptr_t(int ptr) :
        ptr_(ptr)
    {}
    inline accptr_t(long int ptr) :
        ptr_(ptr)
    {}
    inline accptr_t(unsigned long ptr) :
        ptr_(ptr)
    {}

#if CUDA_VERSION < 3020
    inline operator uint32_t() const { return uint32_t(ptr_); }
#else
    inline operator CUdeviceptr() { return ptr_; }
    inline operator CUdeviceptr() const { return ptr_; }
#endif
    inline operator void *() { return (void *)(ptr_); }
    inline operator void *() const { return (void *)(ptr_); }

    inline void *get() const { return (void *)(ptr_); }
};

static inline
bool operator==(const accptr_t &ptr1, const void *ptr2)
{
    return (((const void *)ptr1.ptr_) == ptr2);
}

static inline
bool operator==(const accptr_t &ptr1, int ptr2)
{
    return (int)ptr1.ptr_ == ptr2;
}

static inline
bool operator==(const accptr_t &ptr1, long int ptr2)
{
    return (((long int)ptr1.ptr_) == ptr2);
}

static inline
bool operator!=(const accptr_t &ptr1, const void *ptr2)
{
    return (((const void *)ptr1.ptr_) != ptr2);
}

static inline
bool operator!=(const accptr_t &ptr1, int ptr2)
{
    return (int)ptr1.ptr_ != ptr2;
}

static inline
bool operator!=(const accptr_t &ptr1, long int ptr2)
{
    return (((long int)ptr1.ptr_) != ptr2);
}

static inline
accptr_t operator+(const accptr_t &ptr1, int add)
{
    return accptr_t(ptr1.ptr_ + add);
}

static inline
accptr_t operator+(const accptr_t &ptr1, long int add)
{
    return accptr_t(ptr1.ptr_ + add);
}

static inline
accptr_t operator+(const accptr_t &ptr1, unsigned add)
{
    return accptr_t(ptr1.ptr_ + add);
}

static inline
accptr_t operator+(const accptr_t &ptr1, size_t add)
{
    return accptr_t(ptr1.ptr_ + long(add));
}

static inline
accptr_t operator-(const accptr_t &ptr1, int sub)
{
    return accptr_t(ptr1.ptr_ - sub);
}

static inline
accptr_t operator-(const accptr_t &ptr1, long int add)
{
    return accptr_t(ptr1.ptr_ - add);
}

static inline
accptr_t operator-(const accptr_t &ptr1, unsigned sub)
{
    return accptr_t(ptr1.ptr_ - sub);
}

static inline
accptr_t operator-(const accptr_t &ptr1, size_t sub)
{
    return accptr_t(ptr1.ptr_ - long(sub));
}

#endif
