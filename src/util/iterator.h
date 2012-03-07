/* Copyright (c) 2009-2011 University of Illinois
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

#ifndef GMAC_UTIL_ITERATOR_H_
#define GMAC_UTIL_ITERATOR_H_

#include "config/common.h"

#include <iterator>

namespace __impl { namespace util {

template <typename It>
class iterator_expander
{
protected:
    It it_;
public:
    inline
    iterator_expander(It it) :
        it_(it)
    {
    }

    inline
	iterator_expander &
    operator++()
    {
        ++it_;
        return *this;
    }

    inline
	iterator_expander &
    operator--()
    {
        --it_;
        return *this;
    }

	bool
    operator!=(const It &it) const
    {
        return it_ != it_;
    }

    inline
	bool
    operator==(const iterator_expander &exp) const
    {
        return it_ == exp.it_;
    }

    inline
	bool
    operator!=(const iterator_expander &exp) const
    {
        return it_ != exp.it_;
    }
};

template <typename It>
class iterator_wrap_associative_first :
    public iterator_expander<It>,
    public std::iterator_traits<It>
{
    typedef iterator_expander<It> parent_expander;
public:
    typedef typename It::value_type::first_type value_type;

    iterator_wrap_associative_first(It it) :
        parent_expander(it)
    {
    }

    iterator_wrap_associative_first(parent_expander exp) :
        parent_expander(exp)
    {
    }

    typename It::value_type::first_type &operator*()
    {
        return parent_expander::it_->first;
    }

    const typename It::value_type::first_type &operator*() const
    {
        return parent_expander::it_->first;
    }

    It &base()
    {
        return parent_expander::it_;
    }

    const It &base() const
    {
        return parent_expander::it_;
    }
};

template <typename It>
class iterator_wrap_associative_second :
    public iterator_expander<It>,
    public std::iterator_traits<It>
{
    typedef iterator_expander<It> parent_expander;
public:
    typedef typename It::value_type::second_type value_type;

    iterator_wrap_associative_second(It it) :
        parent_expander(it)
    {
    }

    iterator_wrap_associative_second(parent_expander exp) :
        parent_expander(exp)
    {
    }

    typename It::value_type::second_type &operator*()
    {
        return parent_expander::it_->second;
    }

    const typename It::value_type::second_type &operator*() const
    {
        return parent_expander::it_->second;
    }

    typename It::value_type::second_type *operator->()
    {
        return &parent_expander::it_->second;
    }

    const typename It::value_type::second_type *operator->() const
    {
        return &parent_expander::it_->second;
    }

    It &base()
    {
        return parent_expander::it_;
    }

    const It &base() const
    {
        return parent_expander::it_;
    }
};

}}

#endif
