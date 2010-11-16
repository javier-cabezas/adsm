#include "FileLock.h"

namespace gmac { namespace util {

FileLock::FileLock(const char * fname, const char *_name) :
    ParaverLock(_name)
{
    _file = fopen(fname, "rw");
    ASSERTION(_file != NULL, "Error opening file '%s' for lock", fname);
    _fd = fileno(_file);
}

FileLock::~FileLock()
{
    fclose(_file);
}

}}
