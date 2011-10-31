#ifndef GMAC_CORE_HPE_DESCRIPTOR_IMPL_H_
#define GMAC_CORE_HPE_DESCRIPTOR_IMPL_H_

#include "util/Logger.h"

namespace __impl { namespace hal { namespace cuda {

template <typename K>
inline
descriptor<K>::descriptor(const std::string &name, K key) :
    key_(key), name_(name)
{
    ASSERTION(key_ != NULL);
}

template <typename K>
inline const std::string &
descriptor<K>::get_name() const
{
    return name_;
}

template <typename K>
inline K
descriptor<K>::get_key() const
{
    return key_;
}

}}}

#endif
