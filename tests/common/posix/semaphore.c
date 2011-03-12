#include "semaphore.h"

void sem_init(sem_t *sem, int value)
{
    pthread_cond_init(&sem->cond, NULL);
    pthread_mutex_init(&sem->mutex, NULL);
    sem->value = value;
}

void sem_post(sem_t *sem, int v)
{
    int i;
    pthread_mutex_lock(&sem->mutex);

    sem->value += v;
    for(i = 0; i < sem->value; i++)
        if(sem->value > 0) pthread_cond_signal(&sem->cond);

    pthread_mutex_unlock(&sem->mutex);
}

void sem_wait(sem_t *sem, int v)
{
    pthread_mutex_lock(&sem->mutex);

    sem->value -= v;
    while(sem->value < 0) {
        pthread_cond_wait(&sem->cond, &sem->mutex);
    }

    pthread_mutex_unlock(&sem->mutex);
}

void sem_destroy(sem_t *sem)
{
    pthread_mutex_destroy(&sem->mutex);
    pthread_cond_destroy(&sem->cond);
}
