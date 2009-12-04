#include <os/loader.h>

#include <order.h>
#include <debug.h>
#include <paraver.h>

#include <init.h>
#include <memory/Manager.h>
#include <kernel/Context.h>

#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>

#include <errno.h>

SYM(size_t, __libc_fread, void *, size_t, size_t, FILE *);
SYM(size_t, __libc_fwrite, const void *, size_t, size_t, FILE *);

static void __attribute__((constructor(INTERPOSE))) stdcInit(void)
{
	TRACE("Overloading I/O STDC functions");
	LOAD_SYM(__libc_fread, fread);
	LOAD_SYM(__libc_fwrite, fwrite);
}


/* Standard C library wrappers */

#ifdef __cplusplus
extern "C"
#endif
size_t fread(void *buf, size_t size, size_t nmemb, FILE *stream)
{
    gmac::Context *ctx = manager->owner(buf);

    if(__inGmac() == 1 || ctx == NULL) return __libc_fread(buf, size, nmemb, stream);

	__enterGmac();
	pushState(IORead);

    manager->invalidate(buf, size * nmemb);
    void *tmp = malloc(size * nmemb);

    size_t ret = __libc_fread(tmp, size, nmemb, stream);
    ctx->copyToDevice(manager->ptr(buf), tmp, size * nmemb);
    free(tmp);

    popState();
	__exitGmac();

    return ret;
}


#ifdef __cplusplus
extern "C"
#endif
size_t fwrite(const void *buf, size_t size, size_t nmemb, FILE *stream)
{
    gmac::Context *ctx = manager->owner(buf);

    if(__inGmac() == 1 || ctx == NULL) return __libc_fwrite(buf, size, nmemb, stream);

	__enterGmac();
    pushState(IOWrite);

    void *tmp = malloc(size * nmemb);

    manager->flush(buf, size * nmemb);
    ctx->copyToHost(tmp, manager->ptr(buf), size * nmemb);

    size_t ret =  __libc_fwrite(tmp, size, nmemb, stream);
    free(tmp);
	
    popState();
	__exitGmac();

    return ret;
}
