#ifndef GMAC_UTIL_OBSERVER_IMPL_H_
#define GMAC_UTIL_OBSERVER_IMPL_H_

#include <algorithm>

#include "Logger.h"

namespace __impl { namespace util {

template <typename T>
void
observable<T>::notify()
{
    typename list_observer::iterator it;

    for (it = observers_.begin(); it != observers_.end(); it++) {
        (*it)->update(*(T *)this);
    }
}

template <typename T>
void
observable<T>::add_observer(observer<T> &obj)
{
    ASSERTION(std::find(observers_.begin(), observers_.end(), &obj) == observers_.end());
    observers_.push_back(&obj);
}

}}

#endif
