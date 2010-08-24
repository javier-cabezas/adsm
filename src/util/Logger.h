/* Copyright (c) 2009 University of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#ifndef __UTIL_LOGGER_H_
#define __UTIL_LOGGER_H_

#include <config.h>
#include <threads.h>

#include "Lock.h"
#include "Parameter.h"

#include <map>
#include <list>
#include <string>
#include <iostream>
#include <cstring>
#include <cstdarg>
#include <cassert>

#define ASSERT_STRING "in function %s [%s:%d]", __func__, __FILE__, __LINE__
#define assertion(c, ...) __assertion(c, ASSERT_STRING)
#define ASSERTION(c, ...) __Assertion(c, ASSERT_STRING)

#define SRC_ROOT "src"
inline const char *__extract_file_name(const char *file) {
    return strstr(file, SRC_ROOT);
}

#define trace(fmt, ...) __trace("("FMT_TID":%s) [%s:%d] " fmt, SELF(), __func__, \
    __extract_file_name(__FILE__), __LINE__, ##__VA_ARGS__)
#define TRACE(fmt, ...) __Trace("("FMT_TID":%s) [%s:%d] " fmt, SELF(), __func__, \
    __extract_file_name(__FILE__), __LINE__, ##__VA_ARGS__)

#define WARNING(fmt, ...) __Warning("("FMT_TID")" fmt, SELF(), ##__VA_ARGS__)


namespace gmac { namespace util {


class LoggerLock : public Lock {
protected:
    friend class Logger;
public:
    LoggerLock();
};

class Logger {
private:
    Logger(const char *name);
    void init();

    static Logger *__logger;
protected:
    static Parameter<const char *> *Level;
    static const char *debugString;
    static std::list<std::string> *tags;
    static LoggerLock lock;

    const char *name;
    bool active;
    std::ostream *out;

    static const size_t BufferSize = 1024;
    static char buffer[BufferSize];

    void print(std::string tag, const char *fmt, va_list list) const;
#ifdef DEBUG
    bool check(const char *name) const;
    void log(std::string tag, const char *fmt, va_list list) const;
#endif

    Logger();

    void __trace(const char *fmt, ...) const; 
    void warning(const char *fmt, ...) const;
    void __assertion(unsigned c, const char *fmt, ...) const;
public:
    virtual inline ~Logger() {};

    static void Create(const char *name);
    static void Destroy();

    static void __Trace(const char *fmt, ...);  
    static void __Assertion(unsigned c, const char *fmt, ...);
    static void __Warning(const char *fmt, ...);

    static void fatal(const char *fmt, ...);
    static void cfatal(unsigned c, const char *fmt, ...);
};

}}

#include "Logger.ipp"

#endif
