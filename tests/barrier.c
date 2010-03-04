#include <stdio.h>
#include <string.h>

#include <assert.h>
#include <errno.h>


#include "barrier.h"

#ifndef SEM_R
#define SEM_R S_IRUSR
#endif

#ifndef SEM_A
#define SEM_A S_IWUSR
#endif


void barrier_init(barrier_t *barrier, int value)
{
    pthread_cond_init(&barrier->cond, NULL);
    pthread_mutex_init(&barrier->mutex, NULL);
    barrier->counter = barrier->value = value;
}


void barrier_wait(barrier_t *barrier)
{
    pthread_mutex_lock(&barrier->mutex);

    barrier->counter--;
    if(barrier->counter > 0)
        pthread_cond_wait(&barrier->cond, &barrier->mutex);

    barrier->counter++;
    if(barrier->counter < barrier->value)
        pthread_cond_signal(&barrier->cond);

    pthread_mutex_unlock(&barrier->mutex);
}

void barrier_destroy(barrier_t *barrier)
{
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->cond);
}
