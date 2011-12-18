#ifndef GMAC_MEMORY_BLOCK_IMPL_H_
#define GMAC_MEMORY_BLOCK_IMPL_H_

#include "memory.h"
#ifdef USE_VM
#include "vm/Bitmap.h"
#include "core/Mode.h"
#endif

namespace __impl { namespace memory {

inline
block::block(hostptr_t addr, hostptr_t shadow, size_t size) :
    Lock("block"),
    size_(size),
    addr_(addr),
    shadow_(shadow)
{
}

inline const block::bounds
block::get_bounds() const
{
    // No need for lock -- addr_ is never modified
    return bounds(addr_, addr_ + size_);
}

inline size_t
block::size() const
{
    return size_;
}

inline hostptr_t
block::get_shadow() const
{
    return shadow_;
}

}}

#endif
