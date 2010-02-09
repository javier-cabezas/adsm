#ifndef __SEMAPHORE_IPP_
#define __SEMAPHORE_IPP_
    
inline void
Semaphore::post()
{
    SEM_POST(__sem);
}

inline void
Semaphore::wait()
{
    SEM_WAIT(__sem);
}

#endif
