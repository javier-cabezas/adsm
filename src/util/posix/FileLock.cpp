#include "FileLock.h"

namespace gmac { namespace util {

FileLock::FileLock(const char * fname, paraver::LockName _name) :
    _name(_name)
{
    _file = fopen(fname, "rw");
    assertion(_file != NULL, "Error opening file '%s' for lock", fname);
    _fd = fileno(_file);
}

FileLock::~FileLock()
{
    fclose(_file);
}

}}
