#include "config/common.h"
#include "memory/address_space.h"

#include "slab.h"

namespace __impl { namespace memory { namespace allocator {

slab::~slab()
{
    map_aspace::iterator i;
    aspaces_.lock_write();
    for(i = aspaces_.begin(); i != aspaces_.end(); ++i) {
        map_cache &map = i->second;
        map_cache::iterator j;
        for(j = map.begin(); j != map.end(); ++j) {
            delete j->second;
        }
        map.clear();
    }
    aspaces_.clear();
    aspaces_.unlock();
}

cache &
slab::create_cache(address_space_ptr aspace, map_cache &map, long_t key, size_t size)
{
    cache *cache = new __impl::memory::allocator::cache(manager_, aspace, size);
    std::pair<map_cache::iterator, bool> ret = map.insert(map_cache::value_type(key, cache));
    ASSERTION(ret.second == true);
    return *cache;
}

cache &
slab::get(address_space_ptr current, long_t key, size_t size)
{
    map_cache *map = NULL;
    map_aspace::iterator i;
    aspaces_.lock_read();
    i = aspaces_.find(current);
    if(i != aspaces_.end()) map = &(i->second);
    aspaces_.unlock();
    if(map == NULL) {
        aspaces_.lock_write();
        cache &ret = create_cache(current, aspaces_[current], key, size);
        aspaces_.unlock();
        return ret;
    }
    else {
        map_cache::iterator j;
        j = map->find(key);
        if(j == map->end())
            return create_cache(current, *map, key, size);
        else
            return *j->second;
    }
}

void
slab::cleanup(address_space_ptr current)
{
    map_aspace::iterator i;
    aspaces_.lock_read();
    i = aspaces_.find(current);
    aspaces_.unlock();
    if(i == aspaces_.end()) return;
    map_cache::iterator j;
    for(j = i->second.begin(); j != i->second.end(); ++j) {
        delete j->second;
    }
    i->second.clear();
    aspaces_.lock_write();
    aspaces_.erase(i);
    aspaces_.unlock();
}

host_ptr
slab::alloc(address_space_ptr current, size_t size, host_const_ptr addr)
{
    cache &cache = get(current, long_t(addr), size);
    TRACE(LOCAL,"Using cache %p", &cache);
    host_ptr ret = cache.get();
    if(ret == NULL) return NULL;
    addresses_.lock_write();
    addresses_.insert(map_address::value_type(ret, &cache));
    addresses_.unlock();
    TRACE(LOCAL,"Returning address %p", ret);
    return ret;
}

bool
slab::free(address_space_ptr current, host_ptr addr)
{
    addresses_.lock_write();
    map_address::iterator i = addresses_.find(addr);
    if(i == addresses_.end()) {
        addresses_.unlock();
        TRACE(LOCAL,"%p was not delivered by slab allocator", addr); 
        return false;
    }
    TRACE(LOCAL,"Inserting %p in cache %p", addr, i->second);
    cache &cache = *(i->second);
    addresses_.erase(i);
    addresses_.unlock();
    cache.put(addr);
    return true;
}

}}}
