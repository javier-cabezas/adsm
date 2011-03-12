#ifndef GMAC_TESTS_COMMON_SEMAPHORE_H_
#define GMAC_TESTS_COMMON_SEMAPHORE_H_

#if defined(POSIX)
#include "posix/semaphore.h"
#elif defined(WINDOWS)
#include "windows/semaphore.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sem_init(sem_t *, int);
void sem_post(sem_t *, int );
void sem_wait(sem_t *, int );
void sem_destroy(sem_t *);


#ifdef __cplusplus
}
#endif

#endif
