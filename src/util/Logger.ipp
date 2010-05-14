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
void Logger::__assertion(unsigned c, const char *fmt, ...) const
{
#ifdef DEBUG
    if(c) return;
    va_list list;
    va_start(list, fmt);
    log("ASSERT", fmt, list);
    va_end(list);
    assert(c);
#endif
}

inline
void Logger::fatal(const char *fmt, ...) const
{
    va_list list;
    va_start(list, fmt);
    log("FATAL", fmt, list);
    va_end(list);
    assert(0);
}

inline
void Logger::warning(const char *fmt, ...) const
{
    va_list list;
    va_start(list, fmt);
    log("WARNING", fmt, list);
    va_end(list);
}


inline
void Logger::cfatal(unsigned c, const char *fmt, ...) const
{
    if(c) return;
    va_list list;
    va_start(list, fmt);
    log("FATAL", fmt, list);
    va_end(list);
    assert(0);
}

}}
#endif
