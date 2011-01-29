#ifndef GMAC_CONFIG_OPENCL_COMMON_H_
#define GMAC_CONFIG_OPENCL_COMMON_H_

#include <CL/cl.h>

#include <cassert>
#include <cstdlib>

struct _opencl_ptr_t {
    cl_mem base_;
    size_t offset_;

    inline _opencl_ptr_t() :
        base_(0),
        offset_(0)
    { }

    inline _opencl_ptr_t(cl_mem base) :
        base_(base),
        offset_(0)
    {}

    inline _opencl_ptr_t(cl_mem base, size_t offset) :
        base_(base),
        offset_(offset)
    { }
    inline _opencl_ptr_t(const _opencl_ptr_t &ptr) :
        base_(ptr.base_),
        offset_(ptr.offset_)
    { }

    inline const _opencl_ptr_t & operator=(const _opencl_ptr_t &ptr) {
        base_   = ptr.base_;
        offset_ = ptr.offset_;
        return *this;
    }

    template <typename T>
    inline const _opencl_ptr_t operator+(const T off) const {
        _opencl_ptr_t tmp;
        tmp.base_   = base_;
        tmp.offset_ = offset_ + off;
        return tmp;
    }

    template <typename T>
    inline const _opencl_ptr_t operator-(const T off) const {
        assert(off < offset_);
        _opencl_ptr_t tmp;
        tmp.base_   = base_;
        tmp.offset_ = offset_ - off;
        return tmp;
    }

    template <typename T>
    inline const bool operator!=(const T addr) const {
        if (addr != T(NULL)) abort();
        return base_ != NULL || offset_ != 0;
    }

    inline const bool operator!=(const _opencl_ptr_t ptr) const {
        return base_ != ptr.base_ || offset_ != ptr.offset_;
    }

    inline operator void*() { return (void *)offset_; }

    inline cl_mem get() const { return base_; }
};

typedef _opencl_ptr_t accptr_t;

#endif
