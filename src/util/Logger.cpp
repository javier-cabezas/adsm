#include "Logger.h"
#include "Lock.h"

#include <fstream>
#include <cstring>
#include <cstdarg>
#include <cassert>

#include <typeinfo>
#if defined(__GNUC__)
#include <cxxabi.h>
#define demangle(name) abi::__cxa_demangle(name, NULL, 0, NULL)
#elif defined(_MSC_VER)
#include <Dbghelp.h>
static char *demangle(const char *name)
{
	char *ret = NULL;
	ret = (char *)malloc(strlen(name));
	if(UnDecorateSymbolName(name, ret, (DWORD)strlen(name), UNDNAME_COMPLETE))
		return ret;
	free(free);
	return NULL;
}
#endif

#if defined(__GNUC__)
#include <strings.h>
#define STRTOK strtok_r
#define VSNPRINTF vsnprintf
#elif defined(_MSC_VER)
#define STRTOK strtok_s
#define VSNPRINTF(str, size, format, ap) vsnprintf_s(str, size, size - 1, format, ap)
static char *strcasestr(const char *haystack, const char *needle)
{
	const char *p, *startn = 0, *np = 0;
	for(p = haystack; *p; p++) {
		if(np) {
			if(toupper(*p) == toupper(*np)) {
				if(!*++np) return (char *)startn;
			}
			else {
				np = 0;
			}
		} else if (toupper(*p) == toupper(*needle)) {
			np = needle + 1;
			startn = p;
		}
	}
	return 0;
}
#endif

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
		char *next = NULL;
		char *tag = STRTOK(tmp, ", ", &next);
        while(tag != NULL) {
            Tags_->push_back(std::string(tag));
            tag = STRTOK(tmp, ", ", &next);
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
    if(name_ == NULL) name  = demangle(typeid(*this).name());
    else name = (char *)name_;

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
    if(name_ == NULL) name  = demangle(typeid(*this).name());
    else name = name_;

    
	VSNPRINTF(Buffer_, BufferSize_, fmt, list);
    *out_ << tag << " [" << name << "]: " << Buffer_ << std::endl;
}

void Logger::print(const char *tag, const char *fmt, va_list list)  const
{
    Lock_.lock();
    __print(tag, fmt, list);
    Lock_.unlock();
}

}}
