#ifndef __MEMORY_CACHEREGION_IPP_
#define __MEMORY_CACHEREGION_IPP_

inline void
RollingRegion::push(RollingBlock *region)
{
    memory.insert(region);
}


inline void
RollingBlock::silentInvalidate()
{
    _present = _dirty = false;
}

inline RollingRegion &
RollingBlock::getParent()
{
    return _parent;
}

#endif
