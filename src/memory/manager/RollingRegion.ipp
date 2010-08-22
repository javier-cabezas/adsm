#ifndef __MEMORY_CACHEREGION_IPP_
#define __MEMORY_CACHEREGION_IPP_

namespace gmac { namespace memory { namespace manager {

inline void
RollingRegion::push(RollingBlock *region)
{
    _memory.lockWrite();
    _memory.insert(region);
    _memory.unlock();
}


inline void
RollingBlock::preInvalidate()
{
   assertion(tryWrite() == false);
   assertion(_dirty == false);
   _present = _dirty = false;
}

inline RollingRegion &
RollingBlock::getParent()
{
    return _parent;
}

inline void
RollingBlock::flush()
{
#ifdef USE_VM
    transfers = 0;
#endif
}

#ifdef USE_VM
inline void
RollingBlock::invalidate()
{
    assertion(tryWrite() == false);
    _present = _dirty = false;
    int ret = Memory::protect(__void(_addr), _size, PROT_NONE);
    assertion(ret == 0);
}

inline void *
RollingBlock::startChunk(unsigned chunk) const
{
    return ((uint8_t *) start()) + chunk * sizeChunk();
}

inline size_t
RollingBlock::sizeChunk() const
{
    return size()/chunks();
}

inline size_t
RollingBlock::chunks() const
{
    return paramBitmapChunksPerPage;
}

inline unsigned
RollingBlock::chunk(const void * addr) const
{
    return (((uint8_t *) addr) - ((uint8_t *) start()))/sizeChunk();
}

inline void
RollingBlock::readOnlyChunk(unsigned chunk)
{
    if (present() == false) _parent.push(this);

    assertion(tryWrite() == false);
    _present = true;
    _dirty = false;
    int ret = Memory::protect(startChunk(chunk), sizeChunk(), PROT_READ);
    assertion(ret == 0);
}

#endif

inline void
RollingBlock::readOnly()
{
    if (present() == false) _parent.push(this);

    assertion(tryWrite() == false);
    _present = true;
    _dirty = false;
    int ret = Memory::protect(__void(_addr), _size, PROT_READ);
    assertion(ret == 0);
}


}}}

#endif
