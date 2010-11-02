#ifndef GMAC_MEMORY_BLOCK_IPP_
#define GMAC_MEMORY_BLOCK_IPP_

namespace gmac { namespace memory { namespace __impl {

inline Block::Block(void *addr, size_t size) :
    util::Lock("memory::Block"),
    addr_(addr),
    size_(size)
{}

inline
Block::~Block()
{}

inline uint8_t *
Block::addr() const
{
    return (uint8_t *) addr_;
}

inline uint8_t *
Block::end() const
{
    return addr() + size_;
}

inline size_t
Block::size() const
{
    return size_;
}

inline void
Block::lock() const
{
    return util::Lock::lock();
}

inline void
Block::unlock() const
{
    return util::Lock::unlock();
}

}}}

#endif
