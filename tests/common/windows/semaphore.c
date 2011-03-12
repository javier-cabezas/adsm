#include "semaphore.h"

void sem_init(sem_t *sem, int value)
{
    InitializeConditionVariable(&sem->cond);
    InitializeCriticalSection(&sem->mutex);
    sem->value = value;
}

void sem_post(sem_t *sem, int v)
{
    int i;
    EnterCriticalSection(&sem->mutex);

    sem->value += v;
    for(i = 0; i < sem->value; i++)
        if(sem->value > 0) WakeConditionVariable(&sem->cond);

    LeaveCriticalSection(&sem->mutex);
}

void sem_wait(sem_t *sem, int v)
{
    EnterCriticalSection(&sem->mutex);

    sem->value -= v;
    while(sem->value < 0) {
        SleepConditionVaraibleCS(&sem->cond, &sem->mutext, INFINITE);
    }

    LeaveCriticalSection(&sem->mutex);
}

void sem_destroy(sem_t *sem)
{
    DeleteCriticalSection(&sem->mutex);
}
