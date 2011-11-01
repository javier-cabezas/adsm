#ifndef GMAC_UTIL_STL_LOCKED_MAP_IMPL_H_
#define GMAC_UTIL_STL_LOCKED_MAP_IMPL_H_

namespace __impl { namespace util { namespace stl {

template <typename K, typename V>
locked_map<K, V>::locked_map(const std::string &name) :
    std::map<K, V>(),
    gmac::util::lock_rw("locked_map")
{
}

template <typename K, typename V>
std::pair<typename locked_map<K, V>::iterator, bool>
locked_map<K, V>::insert(const value_type &x)
{
    lockWrite();
    std::pair<iterator, bool> ret = Parent::insert(x);
    unlock();

    return ret;
}

template <typename K, typename V>
typename locked_map<K, V>::iterator
locked_map<K, V>::insert(iterator position, const value_type &x)
{
    lockWrite();
    iterator ret = Parent::insert(position, x);
    unlock();

    return ret;
}

template <typename K, typename V>
void
locked_map<K, V>::insert(iterator first, iterator last)
{
    lockWrite();
    Parent::insert(first, last);
    unlock();
}

template <typename K, typename V>
inline
typename locked_map<K, V>::iterator
locked_map<K, V>::find(const key_type &key)
{
    lockRead();
    iterator ret = Parent::find(key);
    unlock();

    return ret;
}

#if 0
template <typename K, typename V>
typename locked_map<K, V>::iterator
locked_map<K, V>::begin()
{
    return Parent::begin();
}
#endif

template <typename K, typename V>
inline
typename locked_map<K, V>::iterator
locked_map<K, V>::end()
{
    return Parent::end();
}

template <typename K, typename V>
typename locked_map<K, V>::const_iterator
locked_map<K, V>::find(const key_type &key) const
{
    lockRead();
    const_iterator ret = Parent::find(key);
    unlock();

    return ret;
}

#if 0
template <typename K, typename V>
typename locked_map<K, V>::const_iterator
locked_map<K, V>::begin() const
{
    return Parent::begin();
}
#endif

template <typename K, typename V>
typename locked_map<K, V>::const_iterator
locked_map<K, V>::end() const
{
    return Parent::end();
}

template <typename K, typename V>
void
locked_map<K, V>::erase(iterator position)
{
    lockWrite();
    Parent::erase(position);
    unlock();
}

template <typename K, typename V>
typename locked_map<K, V>::size_type
locked_map<K, V>::erase(const key_type &x)
{
    lockWrite();
    size_type ret = Parent::erase(x);
    unlock();

    return ret;
}

template <typename K, typename V>
void
locked_map<K, V>::erase(iterator first, iterator last)
{
    lockWrite();
    Parent::erase(first, last);
    unlock();
}

template <typename K, typename V>
typename locked_map<K, V>::iterator
locked_map<K, V>::upper_bound(const key_type &x)
{
    lockWrite();
    iterator ret = Parent::upper_bound(x);
    unlock();

    return ret;
}

template <typename K, typename V>
typename locked_map<K, V>::const_iterator
locked_map<K, V>::upper_bound(const key_type &x) const
{
    lockWrite();
    const_iterator ret = Parent::upper_bound(x);
    unlock();

    return ret;
}

template <typename K, typename V>
typename locked_map<K, V>::size_type
locked_map<K, V>::size() const
{
    return Parent::size();
}

}}}

#endif // GMAC_UTIL_STL_LOCKED_MAP_IMPL_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
