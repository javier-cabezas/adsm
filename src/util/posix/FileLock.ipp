#ifndef __UTIL_POSIX_FILE_LOCK_IPP_
#define __UTIL_POSIX_FILE_LOCK_IPP_

#include <sys/file.h>
#include <errno.h>

#include <debug.h>

namespace gmac { namespace util {

inline void
FileLock::lock()
{
    int ret;
    push();
    ret = flock(_fd, LOCK_EX);
    assertion(ret == 0, "Error locking file: %s", strerr(errno));
    pop();
}

inline void
FileLock::unlock()
{
    int ret;
    ret = flock(_fd, LOCK_UN);
    assertion(ret == 0, "Error unlocking file: %s", strerr(errno));
}

inline bool
FileLock::tryLock()
{
    int ret;
    ret = flock(_fd, LOCK_EX | LOCK_NB);
    assertion(ret == 0 || ret == EWOULDBLOCK, "Error trylocking file: %s", strerr(errno));
    return ret == 0;
}

inline FILE *
FileLock::file()
{
    return _file;
}

}}

#endif
