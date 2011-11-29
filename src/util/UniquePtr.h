/* Copyright (c) 2011 University of Illinois
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

#ifndef GMAC_UTIL_UNIQUE_PTR_H_
#define GMAC_UTIL_UNIQUE_PTR_H_

#include <tr1/memory>

/// \todo move this to its own file
template<bool _Cond, typename _Iftrue, typename _Iffalse>
struct conditional
{
	typedef _Iftrue type;
};

template<typename _Iftrue, typename _Iffalse>
struct conditional<false, _Iftrue, _Iffalse>
{
	typedef _Iffalse type;
};

/// \todo move this to its own file
template<bool C1, typename T1,
         bool C2, typename T2,
         bool C3 = false, typename T3 = void,
         bool C4 = false, typename T4 = void,
         bool C5 = false, typename T5 = void>
struct conditional_switch
{
	typedef T1 type;
};

/// \todo move this to its own file
template<typename T1,
         bool C2, typename T2,
         bool C3, typename T3,
         bool C4, typename T4,
         bool C5, typename T5>
struct conditional_switch<true, T1, C2, T2, C3, T3, C4, T4, C5, T5>
{
	typedef T1 type;
};

/// \todo move this to its own file
template<typename T1,
         typename T2,
         bool C3, typename T3,
         bool C4, typename T4,
         bool C5, typename T5>
struct conditional_switch<false, T1, true, T2, C3, T3, C4, T4, C5, T5>
{
	typedef T2 type;
};

/// \todo move this to its own file
template<typename T1,
         typename T2,
         typename T3,
         bool C4, typename T4,
         bool C5, typename T5>
struct conditional_switch<false, T1, false, T2, true, T3, C4, T4, C5, T5>
{
	typedef T3 type;
};

/// \todo move this to its own file
template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         bool C5, typename T5>
struct conditional_switch<false, T1, false, T2, false, T3, true, T4, C5, T5>
{
	typedef T4 type;
};

/// \todo move this to its own file
template<typename T1,
         typename T2,
         typename T3,
         typename T4,
         typename T5>
struct conditional_switch<false, T1, false, T2, false, T3, false, T4, true, T5>
{
	typedef T5 type;
};

template <bool B, class T = void>
struct enable_if_c {
  typedef T type;
};

template <class T>
struct enable_if_c<false, T> {};

template <class Cond, class T = void>
struct enable_if : public enable_if_c<Cond::value, T> {};

namespace __impl { namespace util {
#if 0
    template <typename T>
    struct smart_ptr {
        /*
        typedef std::mierda<T> unique;
        */
        typedef std::tr1::shared_ptr<T> shared;

        template<typename T2>
        static shared
        static_pointer_cast(const std::tr1::shared_ptr<T2> &r)
        {
        	return std::tr1::static_pointer_cast<T>(r);
        }
    };
#endif

    using std::tr1::shared_ptr;

    using std::tr1::is_pointer;
    using std::tr1::is_same;

    using std::tr1::static_pointer_cast;

#if  0
    template <typename T>
    class UniquePtr :
        public std::unique_ptr<T>
    {
    private:
        typedef std::unique_ptr<T> parent;

    public:
        inline
        virtual ~UniquePtr()
        {
        }

        unique_ptr ();
        unique_ptr (
                nullptr_t _Nptr
                );
        explicit unique_ptr (
                pointer _Ptr
                );
        unique_ptr (
                pointer _Ptr,
                typename conditional<
                is_reference<Del>::value, 
                Del,
                typename add_reference<const Del>::type
                >::type _Deleter
                );
        unique_ptr (
                pointer _Ptr,
                typename remove_reference<Del>::type&& _Deleter
                );
        unique_ptr (
                unique_ptr&& _Right
                );
        template<class Type2, Class Del2>
            unique_ptr (
                    unique_ptr<Type2, Del2>&& _Right
                    );

        template<class Y>
        inline
        explicit UniquePtr(Y* p) :
            parent(p)
        {
        }

        template<class Y, class D>
        inline
        UniquePtr(Y* p, D d) :
            parent(p, d)
        {
        }

        inline
        UniquePtr(UniquePtr const& r) :
            parent(r)
        {
        }

        template<class Y>
        inline
        UniquePtr(UniquePtr<Y> const& r) :
            parent(r)
        {
        }

        inline
        UniquePtr& operator=(UniquePtr const& r)
        {
            parent::operator=(r);
            return *this;
        }

        template<class Y> 
        inline
        UniquePtr& operator=(UniquePtr<Y> const& r)
        {
            parent::operator=(r);
            return *this;
        }
    };
#endif

	template <typename T>
	struct is_shared_ptr
	{
		const static bool value = false;
	};

	template <typename T>
	struct is_shared_ptr<shared_ptr<T> >
	{
		const static bool value = true;
	};

	template <typename T>
	struct is_shared_ptr<shared_ptr<const T> >
	{
		const static bool value = true;
	};

	template <typename T>
	struct is_shared_ptr<const shared_ptr<T> >
	{
		const static bool value = true;
	};

	template <typename T>
	struct is_shared_ptr<const shared_ptr<const T> >
	{
		const static bool value = true;
	};

	template <typename T>
	struct is_any_ptr
	{
		const static bool value = is_pointer<T>::value || is_shared_ptr<T>::value;
	};
}}

#endif
