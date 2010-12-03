#ifndef GMAC_MEMORY_DBC_BLOCK_IMPL_H_
#define GMAC_MEMORY_DBC_BLOCK_IMPL_H_

namespace __dbc { namespace memory {

inline
Block::Block(__impl::memory::Protocol &protocol, uint8_t *addr, uint8_t *shadow, size_t size) :
    __impl::memory::Block(protocol, addr, shadow, size)
{
}

inline
Block::~Block()
{
}

}}

#endif /* BLOCK_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
