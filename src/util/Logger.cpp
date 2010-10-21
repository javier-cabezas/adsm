#include "Logger.h"
#include "Lock.h"

#include <fstream>
#include <cstring>
#include <cstdarg>
#include <cassert>

#include <typeinfo>
#include <cxxabi.h>

namespace gmac { namespace util {

char Logger::Buffer_[Logger::BufferSize_];
LoggerLock Logger::Lock_;

Logger *Logger::Logger_ = NULL;

#ifdef DEBUG
Parameter<const char *> *Logger::Level_ = NULL;
const char *Logger::DebugString_;
std::list<std::string> *Logger::Tags_ = NULL;
#endif

LoggerLock::LoggerLock() :
    Lock("Logger")
{}

Logger::Logger(const char *name) :
    name_(name),
    active_(false),
    out_(&std::clog)
{
    init();
}

Logger::Logger() :
    name_(NULL),
    active_(false),
    out_(&std::clog)
{
    init();
}

void Logger::init()
{
#ifdef DEBUG
    if(Tags_ == NULL) Tags_ = new std::list<std::string>();
    if(Level_ == NULL) {
        Level_ = new Parameter<const char *>(&Logger::DebugString_,
            "Logger::DebugString_", "none", "GMAC_DEBUG");
        char *tmp = new char[strlen(DebugString_) + 1];
        memcpy(tmp, DebugString_, strlen(DebugString_) + 1);
        char *tag = strtok(tmp, ", ");
        while(tag != NULL) {
            Tags_->push_back(std::string(tag));
            tag = strtok(NULL, ", ");
        }
        delete[] tmp;
    }

    if(DebugString_ != NULL && strcasestr(DebugString_, "__all") != NULL)
        active_ = true;
#endif
}

#ifdef DEBUG
bool Logger::check(const char *name) const
{
    if(Tags_ == NULL) return false;
    std::list<std::string>::const_iterator i;
    for(i = Tags_->begin(); i != Tags_->end(); i++) {
        if(strstr(name, i->c_str()) != NULL) return true;
    }
    return false;
}

void Logger::log(const char *tag, const char *fmt, va_list list) const
{
    char *name = NULL;
    int status = 0;
    if(name_ == NULL) name  = abi::__cxa_demangle(typeid(*this).name(), NULL, 0, &status);
    else name = (char *)name_;

    if(status != 0) return;

    if(active_ == false && check(name) == false) {
        if(name_ == NULL && name != NULL ) free(name);
        return;
    }

    print(tag, fmt, list);

    if(name_ == NULL && name != NULL ) free(name);
}
#endif

void Logger::__print(const char *tag, const char *fmt, va_list list)  const
{
    const char *name = NULL;
    if(name_ == NULL) name  = abi::__cxa_demangle(typeid(*this).name(), NULL, 0, NULL);
    else name = name_;

    vsnprintf(Buffer_, BufferSize_, fmt, list);
    *out_ << tag << " [" << name << "]: " << Buffer_ << std::endl;
}

void Logger::print(const char *tag, const char *fmt, va_list list)  const
{
    Lock_.lock();
    __print(tag, fmt, list);
    Lock_.unlock();
}

}}
