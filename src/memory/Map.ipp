#ifndef __MEMORY_MAP_IPP_
#define __MEMORY_MAP_IPP_

//#include <memory/Object.h>

namespace gmac { namespace memory {

ObjectMap::ObjectMap(paraver::LockName name) :
    RWLock(name)
{
}


Map::Map(paraver::LockName name) :
    ObjectMap(name)
{ }

Map::~Map()
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

inline Object *
Map::find(const void *addr)
{
    Object *ret = NULL;
    lockRead();
    ret = localFind(addr);
    if(ret == NULL) {
        ret = globalFind(addr);
    }
    if(ret == NULL) {
        ret = sharedFind(addr);
    }
    unlock();

    return ret;
}

}}

#endif
