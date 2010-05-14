#include "Logger.h"
#include "Lock.h"

#include <paraver.h>

#include <fstream>
#include <cstdarg>
#include <cassert>

namespace gmac { namespace util {

const char *Logger::debugString;
Parameter<const char *> *Logger::Level = NULL;
Lock Logger::lock(LockLog);

Logger::Logger(const char *name) :
    name(name),
    active(false),
    out(std::clog)
{
    if(Level == NULL) {
        Level = new Parameter<const char *>(&Logger::debugString,
            "Logger::debugString", "none", "GMAC_DEBUG");
    }
#if 0
    if(paramDebugFile != NULL)
        out = std::ofstream(paramDebugFile);
#endif
    if(debugString != NULL && 
      (strcasestr(debugString, "all") != NULL ||
       strcasestr(debugString, name) != NULL)) {
        active = true;
    }
}

void Logger::log(std::string tag, const char *fmt, va_list list)
{
    vsnprintf(buffer, BufferSize, fmt, list);
    if(active == false) return;
    lock.lock();
    out << tag << " [" << name << "]: " << buffer << std::endl;
    lock.unlock();
}

}}
