#include "Logger.h"
#include "Lock.h"

#include <paraver.h>

#include <fstream>
#include <cstring>
#include <cstdarg>
#include <cassert>

#include <typeinfo>
#include <cxxabi.h>

namespace gmac { namespace util {

char Logger::buffer[Logger::BufferSize];
LoggerLock Logger::__lock;

Logger *Logger::__logger = NULL;

#ifdef DEBUG
Parameter<const char *> *Logger::Level = NULL;
const char *Logger::debugString;
std::list<std::string> *Logger::tags = NULL;
#endif

LoggerLock::LoggerLock() :
    Lock(LockLog)
{}

Logger::Logger(const char *name) :
    name(name),
    active(false),
    out(&std::clog)
{
    init();
}

Logger::Logger() :
    name(NULL),
    active(false),
    out(&std::clog)
{
    init();
}

void Logger::init()
{
#ifdef DEBUG
    if(tags == NULL) tags = new std::list<std::string>();
    if(Level == NULL) {
        Level = new Parameter<const char *>(&Logger::debugString,
            "Logger::debugString", "none", "GMAC_DEBUG");
        char *tmp = new char[strlen(debugString) + 1];
        memcpy(tmp, debugString, strlen(debugString) + 1);
        char *tag = strtok(tmp, ", ");
        while(tag != NULL) {
            tags->push_back(std::string(tag));
            tag = strtok(NULL, ", ");
        }
        delete[] tmp;
    }

    if(debugString != NULL && strcasestr(debugString, "__all") != NULL)
        active = true;
#endif
}

#ifdef DEBUG
bool Logger::check(const char *name) const
{
    if(tags == NULL) return false;
    std::list<std::string>::const_iterator i;
    for(i = tags->begin(); i != tags->end(); i++) {
        if(strstr(name, i->c_str()) != NULL) return true;
    }
    return false;
}

void Logger::log(const char *tag, const char *fmt, va_list list) const
{
    char *__name = NULL;
    int status = 0;
    if(name == NULL) __name  = abi::__cxa_demangle(typeid(*this).name(), NULL, 0, &status);
    else __name = (char *)name;

    if(status != 0) return;

    if(active == false && check(__name) == false) {
        //if(name == NULL) free(__name);
        return;
    }

    print(tag, fmt, list);

    if(name == NULL) free(__name);
}
#endif

void Logger::print(const char *tag, const char *fmt, va_list list)  const
{
    __lock.lock();
    const char *__name = NULL;
    if(name == NULL) __name  = abi::__cxa_demangle(typeid(*this).name(), NULL, 0, NULL);
    else __name = name;

    vsnprintf(buffer, BufferSize, fmt, list);
    *out << tag << " [" << __name << "]: " << buffer << std::endl;
    __lock.unlock();

}


}}
