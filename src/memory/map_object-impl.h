#ifndef GMAC_MEMORY_OBJECTMAP_IMPL_H_
#define GMAC_MEMORY_OBJECTMAP_IMPL_H_

#include "object.h"

namespace __impl { namespace memory {

inline
gmacError_t
map_object::forEachObject(gmacError_t (object::*f)(void))
{
    iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) {
        gmacError_t ret = (i->second->*f)();
        if(ret != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    unlock();
    return gmacSuccess;
}

template <typename P1>
gmacError_t
map_object::forEachObject(gmacError_t (object::*f)(P1 &), P1 &p1)
{
    const_iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) {
        gmacError_t ret = (i->second->*f)(p1);
        if(ret != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    unlock();
    return gmacSuccess;
}


#ifdef DEBUG
inline
gmacError_t map_object::dumpObjects(const std::string &dir, std::string prefix, protocol::common::Statistic stat) const
{
    lockRead();
    const_iterator i;
    for(i = begin(); i != end(); i++) {
        object &obj = *(i->second);
        std::stringstream name;
        name << dir << prefix << "#" << obj.getId() << "-" << obj.getDumps(stat) << "_" << protocol::common::StatisticName[stat];

        std::ofstream out(name.str().c_str(), std::ios_base::trunc);
        ASSERTION(out.good());
        obj.dump(out, stat);

        out.close();
    }
    unlock();
    return gmacSuccess;
}

inline
gmacError_t map_object::dumpObject(const std::string &dir, std::string prefix, protocol::common::Statistic stat, hostptr_t ptr) const
{
    object *obj = getObject(ptr, 1);
    lockRead();
    ASSERTION(obj != NULL);
    std::stringstream name;
    name << dir << prefix << "#" << obj->getId() << "-" << obj->getDumps(stat) << "_" << protocol::common::StatisticName[stat];

    std::ofstream out(name.str().c_str(), std::ios_base::trunc);
    ASSERTION(out.good());
    obj->dump(out, stat);

    out.close();
    unlock();
    return gmacSuccess;
}
#endif

inline
bool
map_object::hasModifiedObjects() const
{
    lockRead();
    bool ret = modifiedObjects_;
    unlock();
    return ret;
}

inline
void
map_object::invalidateObjects()
{
    lockWrite();
    modifiedObjects_ = false;
    unlock();
}

inline
void
map_object::modifiedObjects_unlocked()
{
    modifiedObjects_ = true;
    releasedObjects_ = false;
}

inline
void
map_object::modifiedObjects()
{
    lockWrite();
    modifiedObjects_unlocked();
    unlock();
}

inline
bool
map_object::releasedObjects() const
{
    lockRead();
    bool ret = releasedObjects_;
    unlock();
    return ret;
}

inline
Protocol &
map_object::getProtocol()
{
    return protocol_;
}

#ifdef USE_VM
inline vm::Bitmap&
map_object::getBitmap()
{
    return bitmap_;
}

inline const vm::Bitmap&
map_object::getBitmap() const
{
    return bitmap_;
}
#endif

}}

#endif /* OBJECTMAP_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
