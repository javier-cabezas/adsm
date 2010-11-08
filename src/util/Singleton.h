/*
 * Singleton.h
 *
 *  Created on: Oct 2, 2010
 *      Author: jcabezas
 */

#ifndef GMAC_UTIL_SINGLETON_H_
#define GMAC_UTIL_SINGLETON_H_

#include "config/common.h"

/*
 *
 */
namespace gmac { namespace util {

template <typename T>
class GMAC_LOCAL Singleton {
private:
	static T *Singleton_;
protected:
	Singleton();
public:
	virtual ~Singleton();

	template <typename U>
	static void create();
	static void destroy();
	static T& getInstance();
};

}}

#include "Singleton.ipp"

#endif /* SINGLETON_H_ */
