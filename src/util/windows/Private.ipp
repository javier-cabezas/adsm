#ifndef __UTIL_WINDOWS_PRIVATE_IPP_
#define __UTIL_WINDOWS_PRIVATE_IPP_

#include <cassert>

namespace gmac { namespace util {

template<typename T>
inline Private<T>::~Private()
{
	if(key_ != TLS_OUT_OF_INDEXES)
		TlsFree(key_);
}

template <typename T>
inline
void Private<T>::init(Private &var)
{
	assert((var.key_ = TlsAlloc()) != TLS_OUT_OF_INDEXES);
}

template <typename T>
inline
void Private<T>::set(const void *value)
{
	TlsSetValue(key_, (LPVOID)value);
}

template <typename T>
inline
T *Private<T>::get()
{
    return static_cast<T *>(TlsGetValue(key_));
}

}}

#endif
