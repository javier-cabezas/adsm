#ifndef GMAC_UTIL_LOGGER_IPP_
#define GMAC_UTIL_LOGGER_IPP_

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
    if(Logger_ != NULL) return;
    Logger_ = new Logger(name);
}

inline
void Logger::Destroy()
{
#ifdef DEBUG
    if(Tags_ != NULL) delete Tags_;
    Tags_ = NULL;
#endif

    if(Logger_ == NULL) return;
    delete Logger_;
    Logger_ = NULL;
}

inline
void Logger::__Trace(const char *fmt, ...)
{
#ifdef DEBUG
    if(Logger_ == NULL) return;
    va_list list;
    va_start(list, fmt);
    Logger_->log("TRACE", fmt, list);
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
    if(Logger_ != NULL) Logger_->print("ASSERT", fmt, list);
    va_end(list);
    assert(c);
#endif
}

inline
void Logger::__Warning(const char *fmt, ...)
{
    va_list list;
    va_start(list, fmt);
    if(Logger_ != NULL) Logger_->print("WARNING", fmt, list);
    va_end(list);
}


inline
void Logger::Fatal(const char *fmt, ...)
{
    if(Logger_ == NULL) return;
    va_list list;
    va_start(list, fmt);
    Logger_->print("FATAL", fmt, list);
    va_end(list);
#ifdef DEBUG
    assert(0);
#else
    exit(-1);
#endif
}


inline
void Logger::__CFatal(unsigned c, const char *fmt, ...)
{
    if(c || Logger_ == NULL) return;
    va_list list;
    va_start(list, fmt);
    Logger_->print("FATAL", fmt, list);
    va_end(list);
#ifdef DEBUG
    assert(0);
#else
    exit(-1);
#endif
}

}}
#endif
