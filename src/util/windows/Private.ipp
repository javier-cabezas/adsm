#ifndef __UTIL_WINDOWS_PRIVATE_IPP_
#define __UTIL_WINDOWS_PRIVATE_IPP_

#include "util/Logger.h"

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
	gmac::util::Logger::ASSERTION((var.key_ = TlsAlloc()) != TLS_OUT_OF_INDEXES);
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
    T* ret = static_cast<T *>(TlsGetValue(key_));
	gmac::util::Logger::ASSERTION(GetLastError() == ERROR_SUCCESS);
	return ret;
}

}}

#endif
