#ifndef __BARRIER_H_
#define __BARRIER_H_

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int value;
    int counter;
    pthread_cond_t cond;
    pthread_mutex_t mutex;
} barrier_t;

void barrier_init(barrier_t *, int);
void barrier_wait(barrier_t *);
void barrier_destroy(barrier_t *);

#ifdef __cplusplus
}
#endif

#endif
