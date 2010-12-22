#ifndef GMAC_CONFIG_OPENCL_COMMON_H_
#define GMAC_CONFIG_OPENCL_COMMON_H_

#include <opencl.h>
class _opencl_ptr_t {
    cl_mem base_;
    size_t offset_;
public:
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

    const _opencl_ptr_t operator+(unsigned off)
    {
        _opencl_ptr_t tmp;
        tmp.base_   = base_;
        tmp.offset_ = offset_ + off;
        return tmp;
    }

    const _opencl_ptr_t operator-(unsigned off)
    {
        ASSERTION(off < offset_);
        _opencl_ptr_t tmp;
        tmp.base_   = base_;
        tmp.offset_ = offset_ - off;
        return tmp;
    }
};

typedef _opencl_ptr_t accptr_t;

#endif
