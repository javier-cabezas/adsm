#ifndef __UTIL_POSIX_PRIVATE_IPP_
#define __UTIL_POSIX_PRIVATE_IPP_

namespace gmac { namespace util {

inline
void Private::init(Private &var)
{
    pthread_key_create(&var._key, NULL);
}

inline
void Private::set(const void *value)
{
    pthread_setspecific(_key, value);
}

inline
void *Private::get()
{
    return pthread_getspecific(_key);
}

}}

#endif
