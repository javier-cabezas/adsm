#ifndef __DESCRIPTOR_KERNEL_IPP_
#define __DESCRIPTOR_KERNEL_IPP_

namespace gmac {

template <typename K>
inline
Descriptor<K>::Descriptor(const char * name, K key) :
    _name(name),
    _key(key)
{
}

template <typename K>
inline
const char *
Descriptor<K>::name() const
{
    return _name;
}

template <typename K>
inline
K
Descriptor<K>::key() const
{
    return _key;
}

}

#endif
