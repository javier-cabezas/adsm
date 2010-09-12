#ifndef __UTIL_LOGGER_IPP_
#define __UTIL_LOGGER_IPP_

namespace gmac { namespace util {

inline
void Logger::__trace(const char *fmt, ...) const
{
#ifdef DEBUG
    va_list list;
    va_start(list, fmt);
    log("TRACE", fmt, list);
    va_end(list);
#endif
}

inline
void Logger::warning(const char *fmt, ...) const
{
    va_list list;
    va_start(list, fmt);
    print("WARNING", fmt, list);
    va_end(list);
}


inline
void Logger::__assertion(unsigned c, const char *fmt, ...) const
{
#ifdef DEBUG
    if(c) return;
    va_list list;
    va_start(list, fmt);
    print("ASSERT", fmt, list);
    va_end(list);
    assert(c);
#endif
}

inline
void Logger::Create(const char *name)
{
    if(__logger != NULL) return;
    __logger = new Logger(name);
}

inline
void Logger::Destroy()
{
#ifdef DEBUG
    if(tags != NULL) delete tags;
    tags = NULL;
#endif

    if(__logger == NULL) return;
    delete __logger;
    __logger = NULL;
}

inline
void Logger::__Trace(const char *fmt, ...)
{
#ifdef DEBUG
    if(__logger == NULL) return;
    va_list list;
    va_start(list, fmt);
    __logger->log("TRACE", fmt, list);
    va_end(list);
#endif
}

inline
void Logger::__Assertion(unsigned c, const char *fmt, ...)
{
#ifdef DEBUG
    if(c) return;
    va_list list;
    va_start(list, fmt);
    if(__logger != NULL) __logger->print("ASSERT", fmt, list);
    va_end(list);
    assert(c);
#endif
}

inline
void Logger::__Warning(const char *fmt, ...)
{
    va_list list;
    va_start(list, fmt);
    if(__logger != NULL) __logger->print("WARNING", fmt, list);
    va_end(list);
}


inline
void Logger::Fatal(const char *fmt, ...)
{
    if(__logger == NULL) return;
    va_list list;
    va_start(list, fmt);
    __logger->print("FATAL", fmt, list);
    va_end(list);
#ifdef DEBUG
    assert(0);
#else
    exit(-1);
#endif
}


inline
void Logger::CFatal(unsigned c, const char *fmt, ...)
{
    if(c || __logger == NULL) return;
    va_list list;
    va_start(list, fmt);
    __logger->print("FATAL", fmt, list);
    va_end(list);
#ifdef DEBUG
    assert(0);
#else
    exit(-1);
#endif
}

}}
#endif
