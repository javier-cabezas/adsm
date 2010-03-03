#ifndef __SEMAPHORE_IPP_
#define __SEMAPHORE_IPP_
    
inline void
Semaphore::post()
{
    struct sembuf buf;
    buf.sem_num = 0;
    buf.sem_op = 1;
    buf.sem_flg = 0;
    semop(__sem, &buf, 1);
}

inline void
Semaphore::wait()
{
    struct sembuf buf;
    buf.sem_num = 0;
    buf.sem_op = -1;
    buf.sem_flg = 0;
    semop(__sem, &buf, 1);
}

#endif
