#ifndef __SEMAPHORE_IPP_
#define __SEMAPHORE_IPP_
    
inline void
Semaphore::post()
{
    pthread_mutex_lock(&_mutex);

    _val++;
    if(_val >= 0)
        pthread_cond_signal(&_cond);

    pthread_mutex_unlock(&_mutex);
}

inline void
Semaphore::wait()
{
    pthread_mutex_lock(&_mutex);

    _val--;
    if(_val < 0)
        pthread_cond_wait(&_cond, &_mutex);

    pthread_mutex_unlock(&_mutex);
}

#endif
