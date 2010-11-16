#ifndef GMAC_UTIL_WINDOWS_SEMAPHORE_IMPL_H_
#define GMAC_UTIL_WINDOWS_SEMAPHORE_IMPL_H_
    
inline void
Semaphore::post()
{
    EnterCriticalSection(&mutex_);

    val_++;
    if(val_ >= 0)
        WakeConditionVariable(&cond_);

    LeaveCriticalSection(&mutex_);
}

inline void
Semaphore::wait()
{
    EnterCriticalSection(&mutex_);

    val_--;
    while(val_ < 0)
		SleepConditionVariableCS(&cond_, &mutex_, INFINITE);

    LeaveCriticalSection(&mutex_);
}

#endif
