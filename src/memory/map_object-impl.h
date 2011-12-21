#ifndef GMAC_MEMORY_OBJECTMAP_IMPL_H_
#define GMAC_MEMORY_OBJECTMAP_IMPL_H_

#include "object.h"

namespace __impl { namespace memory {

inline
hal::event_ptr
map_object::acquire_objects(GmacProtection prot, gmacError_t &err)
{
    hal::event_ptr ret;
    iterator i;
    lock_read();
    for (i = begin(); i != end(); ++i) {
        ret = (*i->second).acquire(prot, err);
        if (err != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    unlock();
    return ret;
}

inline
hal::event_ptr
map_object::release_objects(bool flushDirty, gmacError_t &err)
{
    hal::event_ptr ret;
    iterator i;
    lock_read();
    for (i = begin(); i != end(); ++i) {
        ret = (*i->second).release(flushDirty, err);
        if (err != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    unlock();
    return ret;
}

#ifdef DEBUG
inline gmacError_t
map_object::dumpObjects(const std::string &dir, std::string prefix, protocols::common::Statistic stat) const
{
    lock_read();
    const_iterator i;
    for(i = begin(); i != end(); ++i) {
        object &obj = *(i->second);
        std::stringstream name;
        name << dir << prefix << "#" << obj.get_print_id() << "-" << obj.getDumps(stat) << "_" << protocols::common::StatisticName[stat];

        std::ofstream out(name.str().c_str(), std::ios_base::trunc);
        ASSERTION(out.good());
        obj.dump(out, stat);

        out.close();
    }
    unlock();
    return gmacSuccess;
}

inline
gmacError_t map_object::dumpObject(const std::string &dir, std::string prefix, protocols::common::Statistic stat, host_ptr ptr) const
{
    object_ptr obj = get_object(ptr, 1);
    lock_read();
    ASSERTION(obj != NULL);
    std::stringstream name;
    name << dir << prefix << "#" << obj->get_print_id() << "-" << obj->getDumps(stat) << "_" << protocols::common::StatisticName[stat];

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
map_object::has_modified_objects() const
{
    lock_read();
    bool ret = modifiedObjects_;
    unlock();
    return ret;
}

inline
void
map_object::invalidate_objects()
{
    lock_write();
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
map_object::modified_objects()
{
    lock_write();
    modifiedObjects_unlocked();
    unlock();
}

inline
bool
map_object::released_objects() const
{
    lock_read();
    bool ret = releasedObjects_;
    unlock();
    return ret;
}

inline
protocol &
map_object::get_protocol()
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
