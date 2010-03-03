#ifndef __BARRIER_H_
#define __BARRIER_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifdef LINUX
union semun {
    int val;    /* Value for SETVAL */
    struct semid_ds *buf;    /* Buffer for IPC_STAT, IPC_SET */
    unsigned short *array;  /* Array for GETALL, SETALL */
    struct seminfo *__buf;  /* Buffer for IPC_INFO
    (Linux-specific) */
};

#endif

typedef struct {
    int sem;
    int value;
} barrier_t;

void barrier_init(barrier_t *, int);
void barrier_wait(barrier_t);
void barrier_destroy(barrier_t);

#ifdef __cplusplus
}
#endif

#endif
