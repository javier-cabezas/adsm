#include "Semaphore.h"
#include <fcntl.h>
#include <cassert>

#ifndef SEM_R
#define SEM_R S_IRUSR
#endif

#ifndef SEM_A
#define SEM_A S_IWUSR
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


namespace gmac { namespace util {

Semaphore::Semaphore(unsigned v)
{
    semun un;
    __sem = semget(IPC_PRIVATE, 1, SEM_R | SEM_A);
    assert(__sem >= 0);
    un.val = v;
    semctl(__sem, 0, SETVAL, un);
}

Semaphore::~Semaphore()
{
    semctl(__sem, 0, IPC_RMID);
}

}}
