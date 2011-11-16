/* Copyright (c) 2009, 2011 University of Illinois
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

#ifndef GMAC_UTIL_GMAC_BASE_H_
#define GMAC_UTIL_GMAC_BASE_H_

#include "Atomics.h"
#include "lock.h"
#include "Logger.h"
#include "unique.h"

namespace __impl { namespace util {

template <typename T>
class GMAC_LOCAL debug_memory {
private:
    static Atomic Alloc_;
    static Atomic Free_;

public:
    debug_memory()
    {
        unsigned(AtomicInc(Alloc_));
    }

    virtual ~debug_memory()
    {
        unsigned(AtomicInc(Free_));
    }

    static void report_debug_info()
    {
        printf("Alloc: %u\n", unsigned(Alloc_));
        printf("Free: %u\n", unsigned(Free_));
    }
};

template <typename T>
Atomic debug_memory<T>::Alloc_ = 0;

template <typename T>
Atomic debug_memory<T>::Free_ = 0;

class GMAC_LOCAL named {
private:
    std::string name_;

public:
    named(const std::string &name) :
        name_(name)
    {
    }

    const std::string &get_name() const
    {
        return name_;
    }
};

typedef void (*report_fn)();

#ifdef DEBUG

class GMAC_LOCAL debug :
    public gmac::util::mutex {
    static debug debug_;

    typedef std::map<std::string, report_fn> MapTypes;
    MapTypes mapTypes_;

    void register_type_(const std::string &name, report_fn fn)
    {
        lock();
        mapTypes_.insert(std::map<std::string, report_fn>::value_type(name, fn));
        unlock();
    }

    void dumpInfo_()
    {
        if (config::params::DebugPrintDebugInfo == true) {
            MapTypes::const_iterator it;
            for (it = mapTypes_.begin(); it != mapTypes_.end(); it++) {
                printf("DEBUG INFORMATION FOR CLASS: %s\n", it->first.c_str());
                it->second();
            }
        }
    }
public:
    debug() :
        gmac::util::mutex("debug")
    {
    }

    ~debug()
    {
        dumpInfo_();
    }

    static void register_type(const std::string &name, report_fn fn)
    {
        debug_.register_type_(name, fn);
    }

    static void dumpInfo()
    {
        debug_.dumpInfo_();
    }
};


#endif

template <typename T, typename R = default_id>
class GMAC_LOCAL gmac_base
#ifdef DEBUG
    :
    public unique<T, R>,
    public debug_memory<T>,
    public named
#endif
{
#ifdef DEBUG
private:
    static std::string get_type_name()
    {
        return std::string(get_name_logger(typeid(T).name()));
    }

    static std::string get_tmp_name(const R &id)
    {
        std::stringstream ss;
        ss << std::string(get_type_name()) << "_" << id.print();
        return ss.str();
    }

    static Atomic registered;

public:
    gmac_base() :
        unique<T, R>(),
        named(get_tmp_name(unique<T, R>::get_id()))
    {
        if (AtomicTestAndSet(registered, 0, 1) == true) {
            debug::register_type(get_type_name(), &report_debug_info);
        }
    }

    static void
    report_debug_info()
    {
        debug_memory<T>::report_debug_info();
    }

    gmac_base(const std::string &name) :
        named(name)
    {
    }

    virtual ~gmac_base()
    {
    }
#endif
};

#ifdef DEBUG
template <typename T, typename R>
Atomic gmac_base<T, R>::registered = 0;
#endif

}}


#endif // GMAC_UTIL_GMAC_BASE_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
