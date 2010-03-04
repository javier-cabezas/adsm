#ifndef __UTIL_POSIX_PRIVATE_IPP_
#define __UTIL_POSIX_PRIVATE_IPP_

inline
void Private::init(Private &var)
{
    pthread_key_create(&var.__key, NULL);
}

inline
void Private::set(const void *value)
{
    pthread_setspecific(__key, value);
}

inline
void *Private::get()
{
    return pthread_getspecific(__key);
}

#endif
