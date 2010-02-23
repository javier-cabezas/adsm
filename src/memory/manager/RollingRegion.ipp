#ifndef __MEMORY_CACHEREGION_IPP_
#define __MEMORY_CACHEREGION_IPP_

inline void
RollingRegion::push(RollingBlock *region)
{
    memory.lockWrite();
    memory.insert(region);
    memory.unlock();
}


inline void
RollingBlock::preInvalidate()
{
   ASSERT(tryWrite() == false);
   ASSERT(_dirty == false);
   _present = _dirty = false;
}

inline RollingRegion &
RollingBlock::getParent()
{
    return _parent;
}

#endif
