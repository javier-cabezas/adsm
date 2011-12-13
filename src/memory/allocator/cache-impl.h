#ifndef GMAC_MEMORY_ALLOCATOR_CACHE_IPP_
#define GMAC_MEMORY_ALLOCATOR_CACHE_IPP_

#include "memory/memory.h"

namespace __impl { namespace memory { namespace allocator {

inline
hostptr_t arena::key() const
{
    ASSERTION(ptr_ != NULL);
    return ptr_ + memory::BlockSize_;
}

inline
const ObjectList &arena::objects() const
{
    ASSERTION(ptr_ != NULL);
    return objects_;
}

inline
bool arena::valid() const
{
    return ptr_ != NULL;
}

inline
bool arena::full() const
{
    ASSERTION(ptr_ != NULL);
    return objects_.size() == size_;
}

inline
bool arena::empty() const
{
    ASSERTION(ptr_ != NULL);
    return objects_.empty();
}

inline
hostptr_t arena::get()
{
    ASSERTION(ptr_ != NULL);
    ASSERTION(objects_.empty() == false);
    hostptr_t ret = objects_.front();
    objects_.pop_front();
    TRACE(LOCAL,"Arena %p has "FMT_SIZE" available objects", this, objects_.size());
    return ret;
}

inline
void arena::put(hostptr_t obj)
{
    ASSERTION(ptr_ != NULL);
    objects_.push_back(obj);
}

inline
cache::cache(manager &manager, util::shared_ptr<core::address_space> aspace, size_t size) :
    Lock("cache"),
    objectSize(size),
    arenaSize(memory::BlockSize_),
    manager_(manager),
    aspace_(aspace)
{ }


inline
void cache::put(hostptr_t obj)
{
    lock();
    map_arena::iterator i;
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
