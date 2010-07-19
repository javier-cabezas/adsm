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
Lock Logger::lock(LockLog);

Logger *Logger::__logger = NULL;

#ifdef DEBUG
Parameter<const char *> *Logger::Level = NULL;
const char *Logger::debugString;
std::list<std::string> *Logger::tags = NULL;
#endif

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
        char *tmp = new char[strlen(debugString)];
        memcpy(tmp, debugString, strlen(debugString));
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

void Logger::log(std::string tag, const char *fmt, va_list list) const
{
    const char *__name = NULL;
    if(name == NULL) __name  = abi::__cxa_demangle(typeid(*this).name(), NULL, 0, NULL);
    else __name = name;

    if(active == false && check(__name) == false) return;

    print(tag, fmt, list);
}
#endif

void Logger::print(std::string tag, const char *fmt, va_list list) const
{
    lock.lock();
    const char *__name = NULL;
    if(name == NULL) __name  = abi::__cxa_demangle(typeid(*this).name(), NULL, 0, NULL);
    else __name = name;

    vsnprintf(buffer, BufferSize, fmt, list);
    *out << tag << " [" << __name << "]: " << buffer << std::endl;
    lock.unlock();

}


}}
