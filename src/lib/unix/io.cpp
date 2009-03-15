#include <common/debug.h>

#ifdef PARAVER
#include <gmac/paraver.h>
#endif

#include <dlfcn.h>
#include <unistd.h>

typedef ssize_t (*read_t)(int, void *, size_t);
typedef ssize_t (*write_t)(int, const void *, size_t);

static read_t _read = NULL;
static write_t _write = NULL;

static void __attribute__((constructor)) gmacIOInit(void)
{
	TRACE("I/O Redirection");
	if((_read = (read_t)dlsym(RTLD_NEXT, "read")) == NULL)
		FATAL("Could not find read()");
	if((_write = (write_t)dlsym(RTLD_NEXT, "write")) == NULL)
		FATAL("Could not find write()");
}

ssize_t read(int fd, void *buf, size_t count)
{
	TRACE("read");
	return _read(fd, buf, count);
}

ssize_t write(int fd, const void *buf, size_t count)
{
	TRACE("write");
	return _write(fd, buf, count);
}
