#include "Cache.h"

#include "core/Context.h"
#include "core/Mode.h"
#include "memory/Manager.h"

#include "util/Parameter.h"
#include "util/Private.h"

namespace __impl { namespace memory { namespace allocator {

Arena::Arena(size_t objSize) :
    ptr_(NULL),
    size_(0)
{
    gmacError_t ret = Manager::getInstance().alloc(&ptr_, paramPageSize);
    CFATAL(ret == gmacSuccess, "Unable to allocate memory in the accelerator");
    for(size_t s = 0; s < paramPageSize; s += objSize, size_++) {
        TRACE(LOCAL,"Arena %p pushes %p ("FMT_SIZE" bytes)", this, (void *)(ptr_ + s), objSize);
        objects_.push_back(ptr_ + s);
    }
}

Arena::~Arena()
{
    CFATAL(objects_.size() == size_, "Destroying non-full Arena");
    objects_.clear();
	Manager::getInstance().free(ptr_);
}


Cache::~Cache()
{
    ArenaMap::iterator i;
    for(i = arenas.begin(); i != arenas.end(); i++) {
        delete i->second;
    }
  
}

hostptr_t Cache::get()
{
    ArenaMap::iterator i;
    lock();
    for(i = arenas.begin(); i != arenas.end(); i++) {
        if(i->second->empty()) continue;
        TRACE(LOCAL,"Cache %p gets memory from arena %p", this, i->second);
        unlock();
        return i->second->get();
    }
    // There are no free objects in any arena
    Arena *arena = new __impl::memory::allocator::Arena(objectSize);
    TRACE(LOCAL,"Cache %p creates new arena %p with key %p", this, arena, arena->key());
    arenas.insert(ArenaMap::value_type(arena->key() , arena));
    hostptr_t ptr = arena->get();
    unlock();
    return ptr;
}

}}}
