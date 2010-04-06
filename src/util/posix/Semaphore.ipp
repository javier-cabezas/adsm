#ifndef __SEMAPHORE_IPP_
#define __SEMAPHORE_IPP_
    
inline void
Semaphore::post()
{
    pthread_mutex_lock(&__mutex);

    __val++;
    if(__val >= 0)
        pthread_cond_signal(&__cond);

    pthread_mutex_unlock(&__mutex);
}

inline void
Semaphore::wait()
{
    pthread_mutex_lock(&__mutex);

    __val--;
    if(__val < 0)
        pthread_cond_wait(&__cond, &__mutex);

    pthread_mutex_unlock(&__mutex);
}

#endif
