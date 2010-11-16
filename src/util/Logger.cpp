#include "Logger.h"
#include "Private.h"

#include <fstream>
#include <cstring>
#include <cstdarg>
#include <cassert>

#include <typeinfo>
#if defined(__GNUC__)
#include <cxxabi.h>
#define demangle(name) abi::__cxa_demangle(name, NULL, 0, NULL)
#elif defined(_MSC_VER)
static char *demangle(const char *name)
{
	char *ret = NULL;
	ret = (char *)malloc(strlen(name) + 1);
	memcpy(ret, name, strlen(name) + 1);
	return ret;
}
#endif

#if defined(_MSC_VER)
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

#ifdef DEBUG
bool Logger::Ready_ = false;
Parameter<const char *> *Logger::Level_ = NULL;
const char *Logger::DebugString_;
std::list<std::string> *Logger::Tags_ = NULL;
Private<char> Logger::Buffer_;    
#endif

Logger::Logger() {}

Logger::~Logger() {}

void Logger::Init()
{
#ifdef DEBUG
	Private<char>::init(Buffer_);
	Buffer_.set(new char[BufferSize_]);

    Tags_ = new std::list<std::string>();
    Level_ = new Parameter<const char *>(&Logger::DebugString_, "Logger::DebugString_", "none", "GMAC_DEBUG");
    char *tmp = new char[strlen(DebugString_) + 1];
    memcpy(tmp, DebugString_, strlen(DebugString_) + 1);
	char *next = NULL;
	char *tag = STRTOK(tmp, ", ", &next);
    while(tag != NULL) {
        Tags_->push_back(std::string(tag));
        tag = STRTOK(NULL, ", ", &next);
    }
    delete[] tmp;

	Ready_ = true;
#endif
}

#ifdef DEBUG
bool Logger::Check(const char *name)
{
    if(Ready_ == false) return false;
	if(name == NULL) return true;
    std::list<std::string>::const_iterator i;
    for(i = Tags_->begin(); i != Tags_->end(); i++) {
        if(strstr(name, i->c_str()) != NULL) return true;
    }
    return false;
}

void Logger::Log(const char *name, const char *tag, const char *fmt, va_list list)
{
    if(Check(name) == false) return;
    Print(tag, name, fmt, list);
}
#endif

void Logger::Print(const char *tag, const char *name, const char *fmt, va_list list)
{
	char *buffer = Buffer_.get();
	if(buffer == NULL) {
		buffer = new char[BufferSize_];
		Buffer_.set(buffer);
	}
	
	VSNPRINTF(buffer, BufferSize_, fmt, list);
	if(name != NULL) fprintf(stderr,"%s [%s]: %s\n", tag, name, buffer);
	else fprintf(stderr,"%s: %s\n", tag, buffer);
}

}}
