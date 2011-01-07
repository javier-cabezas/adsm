#ifndef GMAC_CONFIG_OPENCL_COMMON_H_
#define GMAC_CONFIG_OPENCL_COMMON_H_

#include <CL/cl.h>

#include <cassert>

class _opencl_ptr_t {
    cl_mem base_;
    size_t offset_;
public:
    _opencl_ptr_t() :
        base_(0),
        offset_(0)
    {
    }
    _opencl_ptr_t(void *addr) :
        base_(cl_mem(addr)),
        offset_(0)
    {
    }

    _opencl_ptr_t(cl_mem base, size_t offset) :
        base_(base),
        offset_(offset)
    {
    }

    _opencl_ptr_t(const _opencl_ptr_t &ptr) :
        base_(ptr.base_),
        offset_(ptr.offset_)
    {
    }

    const _opencl_ptr_t & operator=(const _opencl_ptr_t &ptr)
    {
        base_   = ptr.base_;
        offset_ = ptr.offset_;
        return *this;
    }

    const _opencl_ptr_t operator+(unsigned off) const
    {
        _opencl_ptr_t tmp;
        tmp.base_   = base_;
        tmp.offset_ = offset_ + off;
        return tmp;
    }

    const _opencl_ptr_t operator-(unsigned off) const
    {
        assert(off < offset_);
        _opencl_ptr_t tmp;
        tmp.base_   = base_;
        tmp.offset_ = offset_ - off;
        return tmp;
    }

    operator void*() const
    {
        return ((void *)offset_);
    }
};

typedef _opencl_ptr_t accptr_t;

#endif
