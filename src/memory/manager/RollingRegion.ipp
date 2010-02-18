#ifndef __MEMORY_CACHEREGION_IPP_
#define __MEMORY_CACHEREGION_IPP_

inline void
RollingRegion::push(ProtSubRegion *region)
{
    memory.insert(region);
}


inline void
ProtSubRegion::silentInvalidate()
{
    _present = _dirty = false;
}

#endif
