#ifndef GMAC_DSM_MAPPING_IMPL_H_
#define GMAC_DSM_MAPPING_IMPL_H_

#include "coherence/block.h"

namespace __impl { namespace dsm {

inline
size_t
mapping::cursor_block::get_bytes_to_next_block() const
{
    return get_block()->get_size() - offLocal_;
}

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

inline
GmacProtection
mapping::get_protection() const
{
    return prot_;
}

inline
int
mapping::get_flags() const
{
    return flags_;
}

template <bool Hex>
void
mapping::print() const
{
    size_t off = 0;
    for (auto b : blocks_) {
        if (Hex) {
            printf("\t-" FMT_ID2 " %p\n", b->get_print_id2(), (void *) off);
        } else {
            printf("\t-" FMT_ID2 " %zd\n", b->get_print_id2(), off);
        }
        off += b->get_size();
    }
}

}}

#endif /* GMAC_DSM_MAPPING_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
