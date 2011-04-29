#ifndef GMAC_UTIL_SINGLETON_IMPL_H_
#define GMAC_UTIL_SINGLETON_IMPL_H_

#include "Singleton.h"
#include "util/Logger.h"

namespace __impl { namespace util {

template<typename T> T *Singleton<T>::Singleton_ = NULL;
template<typename T> bool Singleton<T>::Valid_ = false;

template<typename T>
inline Singleton<T>::Singleton()
{
    CFATAL(Singleton_ == NULL, "Double initialization of singleton class");
    Singleton_ = static_cast<T *>(this);
    Valid_ = true;
}

template <typename T>
inline Singleton<T>::~Singleton()
{
}

template <typename T>
inline void Singleton<T>::destroy()
{
	ASSERTION(Singleton_ != NULL);
    Valid_ = false;
	delete Singleton_;
	Singleton_ = NULL;
}

template<typename T>
inline T &Singleton<T>::getInstance()
{
	ASSERTION(Singleton_ != NULL);
	return *Singleton_;
}

template <typename T>
template <typename S>
inline S& Singleton<T>::getInstance()
{
	ASSERTION(Singleton_ != NULL);
	return dynamic_cast<S &>(*Singleton_);
}

template<typename T>
inline bool Singleton<T>::isValid()
{
    return Valid_;
}

}}

#endif
