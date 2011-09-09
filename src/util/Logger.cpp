#include <fstream>
#include <cstring>
#include <cstdarg>
#include <cassert>

#include "Logger.h"
#include "Private.h"

#include <typeinfo>
#if defined(__GNUC__)
#include <cxxabi.h>
#define demangle(name) abi::__cxa_demangle(name, NULL, 0, NULL)

const char *
get_class_name(const char *mangled)
{
    char nameBuffer[256];
    size_t len = 256; 
    int s; 
    char* p=abi::__cxa_demangle(mangled, nameBuffer, &len, &s); 
    return p;
}


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

namespace __impl { namespace util {

Private<char> Logger::Buffer_;
#ifdef DEBUG
Atomic Logger::Ready_ = 0;
const char *Logger::DebugString_ = NULL;
Logger::Level *Logger::Level_ = NULL;
Logger::Tags *Logger::Tags_ = NULL;
#endif

DESTRUCTOR(fini);
static void fini()
{
    Logger::Fini();
}

void Logger::Init()
{
    Private<char>::init(Buffer_);
	Buffer_.set(new char[BufferSize_]);
#ifdef DEBUG
    Tags_ = new std::list<std::string>();
    Level_ = new Parameter<const char *>(&DebugString_, "Logger::DebugString_", "none", "GMAC_DEBUG");
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

void Logger::Fini()
{
    char *buffer = Buffer_.get();
    if (buffer) delete [] buffer;
    Buffer_.set(NULL);
#ifdef DEBUG
    if(Ready_ == 0) return;
    if (Level_) delete Level_;
    if (Tags_) delete Tags_;
#endif
}

#ifdef DEBUG
bool Logger::Check(const char *name)
{
    if(AtomicTestAndSet(Ready_, 0, 1) == 0) Init();
	if(name == NULL) return true;
    std::list<std::string>::const_iterator i;
    for(i = Tags_->begin(); i != Tags_->end(); i++) {
        if(strstr(name, i->c_str()) != NULL) return true;
    }
    return false;
}

#endif

}}
