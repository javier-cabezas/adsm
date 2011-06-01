#ifndef GMAC_UTIL_POSIX_SEMAPHORE_IMPL_H_
#define GMAC_UTIL_POSIX_SEMAPHORE_IMPL_H_
    
inline void
Semaphore::post()
{
    pthread_mutex_lock(&_mutex);
    _val++;
    pthread_cond_broadcast(&_cond);
    pthread_mutex_unlock(&_mutex);
}

inline void
Semaphore::wait()
{
    pthread_mutex_lock(&_mutex);
    while (_val == 0) {
        pthread_cond_wait(&_cond, &_mutex);
    }
    _val--;
    pthread_mutex_unlock(&_mutex);
}

#endif
