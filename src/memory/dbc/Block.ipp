#ifndef GMAC_MEMORY_TEST_BLOCK_IPP_
#define GMAC_MEMORY_TEST_BLOCK_IPP_

namespace __dbc { namespace memory {

inline
Block::Block(void *addr, size_t size) :
    __impl::memory::Block(addr, size)
{
    REQUIRES(size > 0);
    REQUIRES(addr != NULL);
}

inline
Block::~Block()
{
}

}}

#endif
