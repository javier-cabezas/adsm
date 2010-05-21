#include "Logger.h"
#include "Lock.h"

#include <paraver.h>

#include <typeinfo>
#include <fstream>
#include <cstring>
#include <cstdarg>
#include <cassert>

namespace gmac { namespace util {

const char *Logger::debugString;
char Logger::buffer[Logger::BufferSize];
Parameter<const char *> *Logger::Level = NULL;
Lock Logger::lock(LockLog);

Logger *Logger::__logger = NULL;

Logger::Logger(const char *name) :
    name(name),
    active(false),
    out(&std::clog)
{
    init();
}

Logger::Logger() :
    name(typeid(*this).name()),
    active(false)
{
    init();
}

void Logger::init()
{
    if(Level == NULL) {
        Level = new Parameter<const char *>(&Logger::debugString,
            "Logger::debugString", "none", "GMAC_DEBUG");
    }

    if(debugString != NULL && 
      (strcasestr(debugString, "__all") != NULL ||
       strcasestr(debugString, name) != NULL)) {
        active = true;
    }
}

void Logger::log(std::string tag, const char *fmt, va_list list, bool force) const
{
    if(active == false && force == false) return;
    lock.lock();
    vsnprintf(buffer, BufferSize, fmt, list);
    *out << tag << " [" << name << "]: " << buffer << std::endl;
    lock.unlock();
}

}}
