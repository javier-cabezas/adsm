#ifndef GMAC_UTIL_SINGLETON_IMPL_H_
#define GMAC_UTIL_SINGLETON_IMPL_H_

#include "trace/logger.h"

namespace __impl { namespace util {

template<typename T> T *singleton<T>::Singleton_ = NULL;

template<typename T>
inline singleton<T>::singleton()
{
    CFATAL(Singleton_ == NULL, "Double initialization of singleton class");
    Singleton_ = static_cast<T *>(this);
}

template <typename T>
inline singleton<T>::~singleton()
{
}

template <typename T>
inline void singleton<T>::destroy()
{
    ASSERTION(Singleton_ != NULL);
    delete static_cast<singleton<T> *>(Singleton_);
    Singleton_ = NULL;
}

template <typename T>
inline T *singleton<T>::getInstance()
{
    return Singleton_;
}

}}

#endif
