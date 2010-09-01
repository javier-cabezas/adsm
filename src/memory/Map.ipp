#ifndef __MEMORY_MAP_IPP_
#define __MEMORY_MAP_IPP_

namespace gmac { namespace memory {

inline ObjectMap::ObjectMap(paraver::LockName name) :
    RWLock(name)
{
}


inline Map::Map(paraver::LockName name) :
    ObjectMap(name)
{ }

inline Map::~Map()
{
    trace("Cleaning Memory Map");
    clean();
}


#ifdef USE_VM
inline vm::Bitmap &
Map::dirtyBitmap()
{
    return __dirtyBitmap;
}

inline const vm::Bitmap &
Map::dirtyBitmap() const
{
    return __dirtyBitmap;
}
#endif

inline Object *Map::localFind(const void *addr)
{
    return mapFind(*this, addr);
}


inline Object *Map::find(const void *addr)
{
    Object *ret = NULL;
    ret = localFind(addr);
    if(ret == NULL) { ret = globalFind(addr); }
#ifndef USE_MMAP
    if(ret == NULL) { ret = sharedFind(addr); }
#endif
    return ret;
}

}}

#endif
