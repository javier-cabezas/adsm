#ifndef GMAC_CORE_DESCRIPTOR_IPP_
#define GMAC_CORE_DESCRIPTOR_IPP_

namespace gmac {

template <typename K>
inline
Descriptor<K>::Descriptor(const char * name, K key) :
    key_(key), name_(name)
{
}

template <typename K>
inline const char *
Descriptor<K>::name() const
{
    return name_;
}

template <typename K>
inline K
Descriptor<K>::key() const
{
    return key_;
}

}

#endif
