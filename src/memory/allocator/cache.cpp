#include "cache.h"

#include "core/address_space.h"
#include "memory/manager.h"

#include "util/Parameter.h"
#include "util/private.h"

namespace __impl { namespace memory { namespace allocator {

arena::arena(manager &manager, core::address_space_ptr aspace, size_t objSize) :
    ptr_(NULL),
    size_(0),
    manager_(manager),
    aspace_(aspace)
{
    gmacError_t ret = manager_.alloc(aspace_, &ptr_, memory::BlockSize_);
    if(ret != gmacSuccess) { ptr_ = NULL; return; }
    for(size_t s = 0; s < memory::BlockSize_; s += objSize, ++size_) {
        TRACE(LOCAL,"Arena %p pushes %p ("FMT_SIZE" bytes)", this, (void *)(ptr_ + s), objSize);
        objects_.push_back(ptr_ + s);
    }
}

arena::~arena()
{
    CFATAL(objects_.size() == size_, "Destroying non-full Arena");
    objects_.clear();
    if(ptr_ != NULL) {
        gmacError_t ret = manager_.free(aspace_, ptr_);
        ASSERTION(ret == gmacSuccess);
    }
}

cache::~cache()
{
    map_arena::iterator i;
    for(i = arenas.begin(); i != arenas.end(); ++i) {
        delete i->second;
    }
}

host_ptr cache::get()
{
    map_arena::iterator i;
    lock();
    for(i = arenas.begin(); i != arenas.end(); ++i) {
        if(i->second->empty()) continue;
        TRACE(LOCAL,"Cache %p gets memory from arena %p", this, i->second);
        unlock();
        return i->second->get();
    }
    // There are no free objects in any arena
    arena *a = new arena(manager_, aspace_, objectSize);
    if(a->valid() == false) {
        delete a;
        unlock();
        return NULL;
    }
    TRACE(LOCAL,"Cache %p creates new arena %p with key %p", this, a, a->key());
    arenas.insert(map_arena::value_type(a->key(), a));
    host_ptr ptr = a->get();
    unlock();
    return ptr;
}

}}}
