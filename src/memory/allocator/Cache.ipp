#ifndef __MEMORY_ALLOCATOR_CACHE_IPP__
#define __MEMORY_ALLOCATOR_CACHE_IPP__

namespace gmac { namespace memory { namespace allocator {

inline
void *Arena::address() const
{
    return ptr;
}

inline
const ObjectList &Arena::objects() const
{
    return __objects;
}

inline
bool Arena::full() const
{
    return __objects.size() == size;
}

inline
bool Arena::empty() const
{
    return __objects.empty();
}

inline
void *Arena::get()
{
    assertion(__objects.empty() == false);
    void *ret = __objects.front();
    __objects.pop_front();
    trace("Arena %p has %zd available objects", this, __objects.size());
    return ret;
}

inline
void Arena::put(void *obj)
{
    __objects.push_back(obj);
}

inline
Cache::Cache(size_t size) :
    objectSize(size),
    arenaSize(paramPageSize)
{ }


inline
void Cache::put(void *obj)
{
    void *key = (void *)((unsigned long)obj & ~(paramPageSize - 1));
    ArenaMap::const_iterator i;
    i = arenas.find(key);
    cfatal(i != arenas.end(), "Address for invalid arena");
    i->second->put(obj);
}

}}}

#endif
