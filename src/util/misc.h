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

#ifndef GMAC_UTIL_MISC_H_
#define GMAC_UTIL_MISC_H_

#include <algorithm>
#include <cstddef>
#include <list>

#include "config/common.h"
#include "trace/logger.h"

namespace __impl { namespace util {

template <typename S>
struct bounds {
    S start;
    S end;

    inline
    bounds(S _start, S _end) :
        start(_start),
        end(_end)
    {
        ASSERTION(_end >= _start);
    }

    inline size_t
    get_size() const
    {
        return end - start;
    }

    inline bool
    is_empty() const
    {
        return start == end;
    }
};

template <typename I>
class range {
    const I begin_;
    const I end_;

public:
    typedef I iterator;
    typedef std::list<typename I::value_type> list;

    inline
    range(I _begin, I _end) :
        begin_(_begin),
        end_(_end)
    {
    }

    inline bool
    is_empty() const
    {
        return begin_ == end_;
    }


    inline const I &
    begin() const
    {
        return begin_;
    }

    inline const I &
    end() const
    {
        return end_;
    }

    list
    to_list() const
    {
        list ret;

        for (auto i : *this) {
            ret.push_back(i);
        }

        return ret;
    }
};

namespace algo {

template <typename C, typename F>
inline
void for_each(C &c, F &&f)
{
    std::for_each(c.begin(), c.end(), f);
}

template <typename C, typename F>
inline
bool
has_predicate(C &&c, F &&f)
{
    return std::find_if(c.begin(), c.end(), f) != c.end();
}

template <typename C, typename F>
inline
typename C::iterator
find(C &&c, typename C::value_type &&t)
{
    return std::find(c.begin(), c.end(), t);
}

template <typename C, typename F>
inline
typename C::iterator
find_if(C &&c, F &&f)
{
    return std::find_if(c.begin(), c.end(), f);
}

template <typename C>
inline
typename C::iterator
find_value(C &&c, typename C::value_type &&t)
{
    return std::find_if(c.begin(), c.end(), [&](const typename C::value_type &val) -> bool
                                            {
                                                return val.second == t;
                                            });
}

template <typename C, typename K>
inline
ptrdiff_t
count(C &&c, K &&k)
{
    return ptrdiff_t(std::count(c.begin(), c.end(), k));
}

template <typename C, typename F>
inline
ptrdiff_t
count_if(C &&c, F &&f)
{
    return ptrdiff_t(std::count_if(c.begin(), c.end(), f));
}

}

}}

#endif
