#ifndef GMAC_MEMORY_ALLOCATOR_CACHE_IPP_
#define GMAC_MEMORY_ALLOCATOR_CACHE_IPP_

namespace __impl { namespace memory { namespace allocator {

inline
void *Arena::key() const
{
    return (uint8_t *)ptr + paramPageSize;
}

inline
const ObjectList &Arena::objects() const
{
    return _objects;
}

inline
bool Arena::full() const
{
    return _objects.size() == size;
}

inline
bool Arena::empty() const
{
    return _objects.empty();
}

inline
void *Arena::get()
{
    ASSERTION(_objects.empty() == false);
    void *ret = _objects.front();
    _objects.pop_front();
    TRACE(LOCAL,"Arena %p has "FMT_SIZE" available objects", this, _objects.size());
    return ret;
}

inline
void Arena::put(void *obj)
{
    _objects.push_back(obj);
}

inline
Cache::Cache(size_t size) :
    util::Lock("Cache"),
    objectSize(size),
    arenaSize(paramPageSize)
{ }


inline
void Cache::put(void *obj)
{
    lock();
    ArenaMap::iterator i;
    i = arenas.upper_bound(obj);
    CFATAL(i != arenas.end(), "Address for invalid arena: %p", obj);
    CFATAL(i->second->address() <= obj, "Address for invalid arena: %p", obj);
    i->second->put(obj);
    if(i->second->full()) {
        delete i->second;
        arenas.erase(i);
    }
    unlock();
}

}}}

#endif
