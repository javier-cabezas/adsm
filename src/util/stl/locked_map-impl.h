#ifndef GMAC_UTIL_STL_LOCKED_MAP_IMPL_H_
#define GMAC_UTIL_STL_LOCKED_MAP_IMPL_H_

namespace __impl { namespace util { namespace stl {

template <typename K, typename V>
locked_map<K, V>::locked_map(const std::string &name) :
    std::map<K, V>(),
    Lock("locked_map")
{
}

template <typename K, typename V>
std::pair<typename locked_map<K, V>::iterator, bool>
locked_map<K, V>::insert(const value_type &x)
{
	Lock::lock_write();
    std::pair<iterator, bool> ret = Parent::insert(x);
    Lock::unlock();

    return ret;
}

template <typename K, typename V>
typename locked_map<K, V>::iterator
locked_map<K, V>::insert(iterator position, const value_type &x)
{
	Lock::lock_write();
    iterator ret = Parent::insert(position, x);
    Lock::unlock();

    return ret;
}

template <typename K, typename V>
void
locked_map<K, V>::insert(iterator first, iterator last)
{
	Lock::lockWrite();
    Parent::insert(first, last);
    Lock::unlock();
}

template <typename K, typename V>
inline
typename locked_map<K, V>::iterator
locked_map<K, V>::find(const key_type &key)
{
	Lock::lock_read();
    iterator ret = Parent::find(key);
    Lock::unlock();

    return ret;
}

#if 0
template <typename K, typename V>
typename locked_map<K, V>::iterator
locked_map<K, V>::begin()
{
    return parent::begin();
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
	Lock::lockRead();
    const_iterator ret = Parent::find(key);
    Lock::unlock();

    return ret;
}

#if 0
template <typename K, typename V>
typename locked_map<K, V>::const_iterator
locked_map<K, V>::begin() const
{
    return parent::begin();
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
	Lock::lock_write();
    Parent::erase(position);
    Lock::unlock();
}

template <typename K, typename V>
typename locked_map<K, V>::size_type
locked_map<K, V>::erase(const key_type &x)
{
	Lock::lock_write();
    size_type ret = Parent::erase(x);
    Lock::unlock();

    return ret;
}

template <typename K, typename V>
void
locked_map<K, V>::erase(iterator first, iterator last)
{
	Lock::lock_write();
    Parent::erase(first, last);
    Lock::unlock();
}

template <typename K, typename V>
typename locked_map<K, V>::iterator
locked_map<K, V>::upper_bound(const key_type &x)
{
	Lock::lock_write();
    iterator ret = Parent::upper_bound(x);
    Lock::unlock();

    return ret;
}

template <typename K, typename V>
typename locked_map<K, V>::const_iterator
locked_map<K, V>::upper_bound(const key_type &x) const
{
	Lock::lock_write();
    const_iterator ret = Parent::upper_bound(x);
    Lock::unlock();

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
