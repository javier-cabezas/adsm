#ifndef GMAC_UTIL_SINGLETON_IMPL_H_
#define GMAC_UTIL_SINGLETON_IMPL_H_

#include "Singleton.h"
#include "util/Logger.h"

namespace __impl { namespace util {

template<typename T> T *Singleton<T>::Singleton_ = NULL;

template <typename T>
Singleton<T>::Singleton()
{
}

template <typename T>
Singleton<T>::~Singleton()
{
}

template <typename T>
template <typename S>
void Singleton<T>::create()
{
    ASSERTION(Singleton_ == NULL);
    Singleton_ = new S();
}

template <typename T>
void Singleton<T>::destroy()
{
	ASSERTION(Singleton_ != NULL);
	delete Singleton_;
	Singleton_ = NULL;
}

template<typename T>
T &Singleton<T>::getInstance()
{
	ASSERTION(Singleton_ != NULL);
	return *Singleton_;
}

template <typename T>
template <typename S>
S& Singleton<T>::getInstance()
{
	ASSERTION(Singleton_ != NULL);
	return dynamic_cast<S &>(*Singleton_);
}

}}

#endif
