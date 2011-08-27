#ifndef GMAC_UTIL_LOGGER_IMPL_H_
#define GMAC_UTIL_LOGGER_IMPL_H_


namespace __impl { namespace util {
#ifdef DEBUG
inline void Logger::__Trace(const char *name, const char *fmt, ...)
{
    if(AtomicTestAndSet(Ready_, 0, 1) == 0) Init();

    va_list list;
    va_start(list, fmt);
    Log(name, "TRACE", fmt, list);
    va_end(list);
}

inline void Logger::__Assertion(bool c, const char *cStr, const char *fmt, ...)
{
    if(AtomicTestAndSet(Ready_, 0, 1) == 0) Init();
    if(c == true ) return;
    va_list list;
    va_start(list, fmt);
    Print(cStr, NULL, fmt, list);
    va_end(list);
    abort();
}
#endif

inline void Logger::__Message(const char *fmt, ...)
{
    va_list list;
    va_start(list, fmt);
	VFPRINTF(stdout, fmt, list);
    va_end(list);
}


inline void Logger::__Warning(const char *fmt, ...)
{
    va_list list;
    va_start(list, fmt);
	VFPRINTF(stderr, fmt, list);
    va_end(list);
}


inline void Logger::__Fatal(const char *fmt, ...)
{
    va_list list;
    va_start(list, fmt);
	VFPRINTF(stderr, fmt, list);
    va_end(list);
    abort();
}


inline void Logger::__CFatal(bool c, const char *cStr, const char *fmt, ...)
{
    if(c == true) return;
    va_list list;
    va_start(list, fmt);
	fprintf(stderr,"%s\n", cStr);
    VFPRINTF(stderr, fmt, list);
    va_end(list);
    abort();
}
}}

#endif
