/* Copyright (c) 2009-2011 Universityversity of Illinois
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

#ifndef GMAC_MEMORY_PROTOCOL_COMMON_BLOCKSTATE_H_
#define GMAC_MEMORY_PROTOCOL_COMMON_BLOCKSTATE_H_

#include <ostream>

#include "config/common.h"

#include "hal/types.h"

namespace __impl {
namespace memory { namespace protocols { namespace common {

enum Statistic {
    PAGE_FAULTS_READ              = 0,
    PAGE_FAULTS_WRITE             = 1,
    PAGE_TRANSFERS_TO_ACCELERATOR = 2,
    PAGE_TRANSFERS_TO_HOST        = 3
};
extern const char *StatisticName[];

template <typename T>
class GMAC_LOCAL block_state {
public:
    typedef T protocol_state;
protected:
    T state_;

    unsigned faultsCacheWrite_;
    unsigned faultsCacheRead_;

public:
    block_state(protocol_state state);

    virtual hal::event_ptr sync_to_device(gmacError_t &err) = 0;
    virtual hal::event_ptr sync_to_host(gmacError_t &err) = 0;

    virtual bool is(protocol_state state) const = 0;

    protocol_state get_state() const;
    virtual void set_state(protocol_state state, hostptr_t addr = NULL) = 0;

    unsigned get_faults_cache_write() const;
    unsigned get_faults_cache_read() const;

    void reset_faults_cache_write();
    void reset_faults_cache_read();

    virtual gmacError_t dump(std::ostream &stream, Statistic stat) = 0;
};

}}}}

#include "block_state-impl.h"

#endif /* BLOCKINFO_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
