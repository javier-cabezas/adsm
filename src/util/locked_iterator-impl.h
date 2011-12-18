#ifndef GMAC_UTIL_LOCKED_OBJECT_IMPL_H_
#define GMAC_UTIL_LOCKED_OBJECT_IMPL_H_

namespace __impl { namespace util {

template <typename P, typename I, typename C>
inline
bool
locking_input_iterator<P, I, C>::operator==(const typename C::iterator &it) const
{
	return parent::get_stl_iterator() == it;
}

template <typename P, typename I, typename C>
inline
bool
locking_input_iterator<P, I, C>::operator==(const typename C::const_iterator &it) const
{
	return parent::get_stl_iterator() == it;
}

template <typename P, typename I, typename C>
inline
bool
locking_input_iterator<P, I, C>::operator!=(const typename C::iterator &it) const
{
	return parent::get_stl_iterator() != it;
}

template <typename P, typename I, typename C>
inline
bool
locking_input_iterator<P, I, C>::operator!=(const typename C::const_iterator &it) const
{
	return parent::get_stl_iterator() != it;
}

template <typename P, typename I, typename C>
inline
bool
locking_input_iterator<P, I, C>::operator==(const P &it) const
{
	return parent::get_stl_iterator() == it.get_stl_iterator();
}

template <typename P, typename I, typename C>
inline
bool
locking_input_iterator<P, I, C>::operator!=(const P &it) const
{
	return parent::get_stl_iterator() != it.get_stl_iterator();
}

template <typename P, typename I, typename C>
P
locked_random_access_iterator<P, I, C>::operator+(int off)
{
	I &it = parent::get_stl_iterator();
	const C &c = parent::get_stl_container();

	P ret(it + off, c);

	return ret;
}

template <typename P, typename I, typename C>
P
locked_random_access_iterator<P, I, C>::operator-(int off)
{
	I &it = parent::get_stl_iterator();
	const C &c = parent::get_stl_container();

	P ret(it - off, c);

	return ret;
}

template <typename P, typename I, typename C>
P &
locked_random_access_iterator<P, I, C>::operator+=(int off)
{
	I &it = parent::get_stl_iterator();
	const C &c = parent::get_stl_container();
	P &p = parent::get_iterator_final();

	if (it != c.end()) {
		p.unlock(p.get_element(it));
	}

	it += off;

	if (it != c.end()) {
		p.lock(p.get_element(it));
	}

	return *((P *)this);
}

template <typename P, typename I, typename C>
P &
locked_random_access_iterator<P, I, C>::operator-=(int off)
{
	I &it = parent::get_stl_iterator();
	const C &c = parent::get_stl_container();
	P &p = parent::get_iterator_final();

	if (it != c.end()) {
		p.unlock(p.get_element(it));
	}

	it -= off;

	if (it != c.end()) {
		p.lock(p.get_element(it));
	}

	return *((P *)this);
}

template <typename P, typename I, typename C>
typename C::value_type &
locked_random_access_iterator<P, I, C>::operator[](int index)
{
	I &it = parent::get_stl_iterator();
	const C &c = parent::get_stl_container();
	P &p = parent::get_iterator_final();

	if (it != c.end()) {
		p.unlock(p.get_element(it));
	}

	typename C::value_type &ret = it[index];

	if (it != c.end()) {
		p.lock(p.get_element(it));
	}

	return ret;
}

template <typename I, typename C, typename E>
locking_iterator_base<I, C, E> &
locking_iterator_base<I, C, E>::operator++()
{
	if (it_ != c_.end()) {
		getter::locker_type::unlock(*getter::get_element(it_));
	}

	++it_;

	if (it_ != c_.end()) {
		getter::locker_type::lock(*getter::get_element(it_));
	}

	return *this;
}

template <typename I, typename C, typename E>
locking_iterator_base<I, C, E>
locking_iterator_base<I, C, E>::operator++(int dummy)
{
	locking_iterator_base ret(it_, c_);

	if (it_ != c_.end()) {
		unlock(getter::get_element(it_));
	}

	it_++;

	if (it_ != c_.end()) {
		lock(getter::get_element(it_));
	}

	return ret;
}

template <typename I, typename C, typename E>
locking_iterator_base<I, C, E> &
locking_iterator_base<I, C, E>::operator--()
{
	if (it_ != getter::c_.end()) {
		getter::locker_type::unlock(getter::get_element(it_));
	}

	--it_;

	if (it_ != getter::c_.end()) {
		getter::locker_type::lock(getter::get_element(it_));
	}

	return *this;
}

template <typename I, typename C, typename E>
locking_iterator_base<I, C, E>
locking_iterator_base<I, C, E>::operator--(int dummy)
{
	locking_iterator_base ret(it_, c_);

	if (it_ != c_.end()) {
		getter::locker_type::unlock(getter::get_element(it_));
	}

	it_--;

	if (it_ != getter::c_.end()) {
		getter::locker_type::lock(getter::get_element(it_));
	}

	return ret;
}

}}

#endif /* GMAC_UTIL_LOCKED_OBJECT_IMPL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
