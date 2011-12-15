/* Copyright (c) 2009-2011sity of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#ifndef GMAC_UTIL_LOCKED_ITERATOR_H_
#define GMAC_UTIL_LOCKED_ITERATOR_H_

#include <utility>

#include "smart_ptr.h"
#include "lock.h"

namespace __impl { namespace util {

//
// Helper classes for conditional inheritance
//
template <typename I>
struct is_random_access_iterator {
    static const bool value = __impl::util::is_same<typename I::iterator_category, std::random_access_iterator_tag>::value;
};

template <typename I>
struct is_bidirectional_iterator {
    static const bool value = __impl::util::is_same<typename I::iterator_category,
                                                    std::bidirectional_iterator_tag>::value ||
                                                    is_random_access_iterator<I>::value;
};

template <typename I>
struct is_forward_iterator {
    static const bool value = __impl::util::is_same<typename I::iterator_category,
                                                    std::forward_iterator_tag>::value   ||
                                                    is_bidirectional_iterator<I>::value ||
                                                    is_random_access_iterator<I>::value;
};

template <typename I>
struct is_output_iterator {
    static const bool value = __impl::util::is_same<typename I::iterator_category,
                                                    std::output_iterator_tag>::value    ||
                                                    is_forward_iterator<I>::value       ||
                                                    is_bidirectional_iterator<I>::value ||
                                                    is_random_access_iterator<I>::value;
};

template <typename I>
struct is_input_iterator {
    static const bool value = __impl::util::is_same<typename I::iterator_category,
                                                    std::input_iterator_tag>::value     ||
                                                    is_forward_iterator<I>::value       ||
                                                    is_bidirectional_iterator<I>::value ||
                                                    is_random_access_iterator<I>::value;
};

template <typename I, typename C>
class _locking_iterator_base {
public:
	typedef typename C::value_type::locker_type locker_type;

	static typename C::value_type *
	get_element(I &it)
	{
		return &*it;
	}
};

template <typename I, typename C>
class _locking_iterator_base_ptr {
public:
	typedef typename C::value_type::element_type::locker_type locker_type;

	static typename C::value_type
	get_element(I &it)
	{
		return *it;
	}
};

#if 0
static const int Input  = 0;
static const int Output = 1;
#endif
template <typename P, typename I, typename C>
class locking_base_iterator {
protected:
	P &get_iterator_final()
	{
		return *((P *) this);
	}

	const P &get_iterator_final() const
	{
		return *((const P *) this);
	}

	I &get_stl_iterator()
	{
		return get_iterator_final().it_;
	}

	const I &get_stl_iterator() const
	{
		return get_iterator_final().it_;
	}

	const C &get_stl_container() const
	{
		return get_iterator_final().c_;
	}
};

template <typename P, typename I, typename C>
class locking_input_iterator :
	public locking_base_iterator<P, I, C> {
	typedef locking_base_iterator<P, I, C> parent;
public:
	bool operator==(const typename C::iterator &it) const;
	bool operator==(const typename C::const_iterator &it) const;
	bool operator!=(const typename C::iterator &it) const;
	bool operator!=(const typename C::const_iterator &it) const;

	bool operator==(const P &it) const;
	bool operator!=(const P &it) const;
};

template <typename P, typename I, typename C>
class locking_output_iterator {
private:
protected:
};

template <typename P, typename I, typename C>
class locking_forward_iterator :
	public locking_input_iterator<P, I, C>,
	public locking_output_iterator<P, I, C> {

	typedef locking_input_iterator<P, I, C> parent_in;
	typedef locking_output_iterator<P, I, C> parent_out;

protected:
};

template <typename P, typename I, typename C>
class locking_bidirectional_iterator :
	public locking_forward_iterator<P, I, C> {
	typedef locking_forward_iterator<P, I, C> parent;
protected:
};

template <typename P, typename I, typename C>
class locked_random_access_iterator :
	public locking_bidirectional_iterator<P, I, C> {

	typedef locking_bidirectional_iterator<P, I, C> parent;

protected:
public:
	inline
	bool operator<(const typename C::iterator &it) const
	{
		return parent::get_stl_iterator() < it;
	}

	inline
	bool operator<(const typename C::const_iterator &it) const
	{
		return parent::get_stl_iterator() < it;
	}

	inline
	bool operator<=(const typename C::iterator &it) const
	{
		return parent::get_stl_iterator() <= it;
	}

	inline
	bool operator<=(const typename C::const_iterator &it) const
	{
		return parent::get_stl_iterator() <= it;
	}

	inline
	bool operator>(const typename C::iterator &it) const
	{
		return parent::get_stl_iterator() > it;
	}

	inline
	bool operator>(const typename C::const_iterator &it) const
	{
		return parent::get_stl_iterator() > it;
	}

	inline
	bool operator>=(const typename C::iterator &it) const
	{
		return parent::get_stl_iterator() >= it;
	}

	inline
	bool operator>=(const typename C::const_iterator &it) const
	{
		return parent::get_stl_iterator() >= it;
	}

	inline
	bool operator<(const P &it) const
	{
		return parent::get_stl_iterator() < it.get_stl_iterator();
	}

	inline
	bool operator<=(const P &it) const
	{
		return parent::get_stl_iterator() <= it.get_stl_iterator();
	}

	inline
	bool operator>(const P &it) const
	{
		return parent::get_stl_iterator() > it.get_stl_iterator();
	}

	inline
	bool operator>=(const P &it) const
	{
		return parent::get_stl_iterator() >= it.get_stl_iterator();
	}

	P operator+(int off);
	P operator-(int off);

	P &operator+=(int off);
	P &operator-=(int off);

	typename C::value_type &operator[](int index);
};

template <typename I, typename C, typename E>
class locking_iterator_base :
	protected conditional<__impl::util::is_any_ptr<E>::value,
		                  _locking_iterator_base_ptr<I, C>,
		                  _locking_iterator_base<I, C> >::type::locker_type,
	public conditional_switch<is_random_access_iterator<I>::value, locked_random_access_iterator<locking_iterator_base<I, C, E>, I, C>,
							  is_bidirectional_iterator<I>::value, locking_bidirectional_iterator<locking_iterator_base<I, C, E>, I, C>,
							  is_forward_iterator<I>::value,       locking_forward_iterator<locking_iterator_base<I, C, E>, I, C>,
							  is_output_iterator<I>::value,        locking_output_iterator<locking_iterator_base<I, C, E>, I, C>,
							  is_input_iterator<I>::value,         locking_input_iterator<locking_iterator_base<I, C, E>, I, C> >::type,
	protected I {
	friend class locking_base_iterator<locking_iterator_base, I, C>;

protected:
	typedef typename conditional<__impl::util::is_any_ptr<E>::value,
				                 _locking_iterator_base_ptr<I, C>,
				                 _locking_iterator_base<I, C> >::type getter;
	typedef typename conditional_switch<is_random_access_iterator<I>::value, locked_random_access_iterator<locking_iterator_base, I, C>,
			                            is_bidirectional_iterator<I>::value, locking_bidirectional_iterator<locking_iterator_base, I, C>,
			                            is_forward_iterator<I>::value,       locking_forward_iterator<locking_iterator_base, I, C>,
			                            is_output_iterator<I>::value,        locking_output_iterator<locking_iterator_base, I, C>,
			                            is_input_iterator<I>::value,         locking_input_iterator<locking_iterator_base, I, C> >::type parent_iterator;

	I it_;
	const C &c_;

	inline
	locking_iterator_base(I &it, const C &c) :
		it_(it),
		c_(c)
	{
		CFATAL(it != c.end(), "Cannot initialize an iterator with an end value");
		lock(*getter::get_element(it_));
	}

	locking_iterator_base(locking_iterator_base &&it) :
		it_(it.it_),
		c_(it.c_)
	{
		// Make old reference point to end to avoid it to unlock the object
		it.it_ = c_.end();
	}

	// These operations are not public since they can cause a double-lock
	locking_iterator_base operator++(int dummy);
	locking_iterator_base operator--(int dummy);

private:
	// Do not allow copies
	locking_iterator_base &operator=(const locking_iterator_base &it);
	locking_iterator_base(const locking_iterator_base &it);

public:
	typedef typename I::iterator_category iterator_category;
	typedef typename I::value_type        value_type;
	typedef typename I::difference_type   difference_type;
	typedef typename I::pointer           pointer;
	typedef typename I::reference         reference;

	virtual
	inline ~locking_iterator_base()
	{
        typename C::value_type ptr;
		if (it_ != c_.end() && (ptr = getter::get_element(it_))) {
			unlock(*ptr);
		}
	}

	locking_iterator_base &operator++();
	locking_iterator_base &operator--();
};

template <typename C>
class locking_iterator :
    public locking_iterator_base<typename C::iterator, C, typename C::value_type> {

    typedef locking_iterator_base<typename C::iterator, C, typename C::value_type> parent;

private:
    locking_iterator(const locking_iterator &it);

public:
    inline
    locking_iterator(locking_iterator &&it) :
        parent(std::move(it))
    {
    }

    inline
    locking_iterator(typename C::const_iterator p, const C &c) :
        parent(p, c)
    {
    }

    inline
	typename C::value_type *operator->()
	{
		return parent::it_.operator->();
	}

    inline
	typename C::value_type &operator*()
	{
		return parent::it_.operator*();
	}
};

template <typename C>
class const_locking_iterator :
    public locking_iterator_base<typename C::const_iterator, C, typename C::value_type> {

    typedef locking_iterator_base<typename C::const_iterator, C, typename C::value_type> parent;

private:
    const_locking_iterator(const const_locking_iterator &it);

public:
    inline
    const_locking_iterator(const_locking_iterator &&it) :
    	parent(std::move(it))
    {
    }

    inline
	const_locking_iterator(typename C::const_iterator p, const C &c) :
		parent(p, c)
	{
	}

    inline
    const_locking_iterator(typename C::iterator p, const C &c) :
       parent(p, c)
    {
    }

    inline
	const typename C::value_type *
    operator->()
	{
		return parent::it_.operator->();
	}

    inline
	const typename C::value_type &
    operator*()
	{
		return parent::it_.operator*();
	}
};

}}

#include "locked_iterator-impl.h"

#endif /* GMAC_UTIL_LOCKED_ITERATOR_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
