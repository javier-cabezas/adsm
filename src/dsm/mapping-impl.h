#ifndef GMAC_DSM_MAPPING_IMPL_H_
#define GMAC_DSM_MAPPING_IMPL_H_

namespace __impl { namespace dsm {

inline
unsigned
mapping::get_nblocks() const
{
    return unsigned(blocks_.size());
}

inline
mapping::bounds
mapping::get_bounds() const
{
    bounds ret(addr_.get_offset(), addr_.get_offset() + size_);

    return ret;
}

inline
hal::ptr
mapping::get_ptr() const
{
    return addr_;
}

}}

#endif /* GMAC_DSM_MAPPING_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
