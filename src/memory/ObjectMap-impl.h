#ifndef GMAC_MEMORY_OBJECTMAP_IMPL_H_
#define GMAC_MEMORY_OBJECTMAP_IMPL_H_

#include "Object.h"

namespace __impl { namespace memory {

inline
gmacError_t ObjectMap::forEachObject(gmacError_t (Object::*f)(void) const) const
{
    const_iterator i;
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

inline
gmacError_t ObjectMap::forEachObject(gmacError_t (Object::*f)(void))
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
gmacError_t ObjectMap::forEachObject(gmacError_t (Object::*f)(P1 &) const, P1 &p1) const
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



template <typename P1>
gmacError_t ObjectMap::forEachObject(gmacError_t (Object::*f)(P1 &), P1 &p1)
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

inline
gmacError_t ObjectMap::dumpObjects(std::string prefix) const
{
#ifdef DEBUG
    lockRead();
    static const protocol::common::Statistic stats[] = { protocol::common::PAGE_FAULTS,
                                                         protocol::common::PAGE_TRANSFERS_TO_ACCELERATOR,
                                                         protocol::common::PAGE_TRANSFERS_TO_HOST };
    for(unsigned i = 0; i < orderedAllocs_.size(); i++) {
        for (unsigned j = 0; j < 3; j++) {
            std::stringstream name;
            name << prefix << "#" << i << "_" << protocol::common::StatisticName[stats[j]];

            std::ofstream out(name.str().c_str(), std::ios_base::trunc);
            ASSERTION(out.good());
            orderedAllocs_[i]->dump(out, stats[j]);

            out.close();
        }
    }
    unlock();
#endif
    return gmacSuccess;
}

}}

#endif /* OBJECTMAP_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
