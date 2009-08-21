#include <os/loader.h>
#include <kernel/Process.h>
#include <kernel/Context.h>

#include <order.h>
#include <paraver.h>
#include <debug.h>

#include <unistd.h>
#include <dlfcn.h>

#include <errno.h>

SYM(pid_t, __libc_fork);
SYM(int, __libc_clone, int (*)(void *), void *, int, void *, ...);

static void __attribute__((constructor(INTERPOSE))) posixForkInit(void)
{
	TRACE("Overloading POSIX fork");
	LOAD_SYM(__libc_fork, fork);
	LOAD_SYM(__libc_clone, clone);
}


#ifdef __cplusplus
extern "C" {
#endif

pid_t fork()
{
	TRACE("fork");
	pid_t pid = __libc_fork();
	if(pid == 0) proc->clone();
	return pid;
}

int clone(int (*fn)(void *), void *, int, void *, ...)
{
	FATAL("clone() not supported by GMAC");
	return 0;
}

#ifdef __cplusplus
}
#endif
