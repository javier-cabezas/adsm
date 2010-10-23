/*
 * Singleton.cpp
 *
 *  Created on: Oct 2, 2010
 *      Author: jcabezas
 */

#include "Singleton.h"

namespace gmac { namespace util {

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
template <typename U>
void Singleton<T>::create()
{
    gmac::util::Logger::ASSERTION(Singleton_ == NULL);
    Singleton_ = new U();
}

template <typename T>
void Singleton<T>::destroy()
{
	gmac::util::Logger::ASSERTION(Singleton_ != NULL);
	delete Singleton_;
	Singleton_ = NULL;
}

template <typename T>
T& Singleton<T>::getInstance()
{
	gmac::util::Logger::ASSERTION(Singleton_ != NULL);
	return *Singleton_;
}

}}
