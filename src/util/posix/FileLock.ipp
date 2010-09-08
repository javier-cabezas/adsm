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
    enter();
    ret = flock(_fd, LOCK_EX);
    assertion(ret == 0, "Error locking file: %s", strerr(errno));
    locked();
}

inline void
FileLock::unlock()
{
    int ret;
    exit();
    ret = flock(_fd, LOCK_UN);
    assertion(ret == 0, "Error unlocking file: %s", strerr(errno));
}


inline FILE *
FileLock::file()
{
    return _file;
}

}}

#endif
