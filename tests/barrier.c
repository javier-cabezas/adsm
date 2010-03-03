#include <stdio.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <fcntl.h>

#include "barrier.h"

#ifndef SEM_R
#define SEM_R S_IRUSR
#endif

#ifndef SEM_A
#define SEM_A S_IWUSR
#endif

void barrier_init(barrier_t *barrier, int value)
{
    union semun un;

    if((barrier->sem = semget(IPC_PRIVATE, 1, SEM_R | SEM_A)) < 0)
        return;
    barrier->value = -1 * value;
    /* Init semaphore to 0 */
    un.val = 0;
    fprintf(stderr,"Init Semaphore = %d\n", -1 * value);
    if(semctl(barrier->sem, 0, SETVAL, un) >= 0) return;
    else return barrier_destroy(*barrier);
}

void barrier_wait(barrier_t barrier)
{
    struct sembuf buf;

    /* Check the current value */
    int value = semctl(barrier.sem, 0, GETVAL);
    fprintf(stderr,"Pre Semaphore = %d\n", value);

    buf.sem_num = 0;
    buf.sem_flg = 0;
    if(value <= barrier.value) {
        buf.sem_op = -2 * value + 1;
        semop(barrier.sem, &buf, 0);
    }
    else {
        buf.sem_op = -1;
        semop(barrier.sem, &buf, 0);
    }

    value = semctl(barrier.sem, 0, GETVAL);
    fprintf(stderr,"Post Semaphore = %d\n", value);
}

void barrier_destroy(barrier_t barrier)
{
    semctl(barrier.sem, 0, IPC_RMID);
    barrier.sem = -1;
}
