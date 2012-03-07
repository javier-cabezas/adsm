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

#ifndef GMAC_UTIL_UNIQUE_H_
#define GMAC_UTIL_UNIQUE_H_

#include <string>
#include <sstream>
#include <typeinfo>

#include "trace/logger.h"

#include "atomics.h"

namespace __impl { namespace util {

template <typename T>
class GMAC_LOCAL printer {

public:
    virtual T print() const = 0;
};

class GMAC_LOCAL default_id :
    printer<unsigned long> {
    unsigned long val_;

public:
    default_id(unsigned long val);

    unsigned long print() const;

    bool operator==(const default_id &id2)
    {
        return val_ == id2.val_;
    }

    bool operator!=(const default_id &id2)
    {
        return val_ != id2.val_;
    }
};

template <typename T, typename R = default_id>
class GMAC_LOCAL unique {
    static Atomic Count_;

private:
    R id_;
    mutable std::string idPrint_;

public:
    unique();

    R get_id() const;
    std::string get_class_name() const
    {
        static std::string strImpl = "__impl::";
        static std::string strDbc  = "__dbc::";

        std::string name = trace::get_class_name(typeid(T).name());
        size_t pos = name.find(strImpl);
        if (pos != std::string::npos) {
            name = name.replace(pos, strImpl.size(), "");
        }
        pos = name.find(strDbc);
        if (pos != std::string::npos) {
            name = name.replace(pos, strDbc.size(), "");
        }
        return name;
    }
    //unsigned long get_print_id() const;
    const char *get_print_id2() const;
};

class GMAC_LOCAL unique_release {
public:
    inline
    unsigned long get_print_id() const
    {
        return 0;
    }
};


}}

#define FMT_ID2 "%s"

#include "unique-impl.h"

#endif
