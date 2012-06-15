/* Copyright (c) 2009-2012 University of Illinois
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

#ifndef GMAC_DSM_MAPPING_H_
#define GMAC_DSM_MAPPING_H_

#include "util/factory.h"
#include "util/misc.h"
#include "util/unique.h"

#include "coherence.h"

#include "error.h"
#include "types.h"

static inline
bool prot_is_writable(GmacProtection prot)
{
    return prot | GMAC_PROT_WRITE;
}

static inline
bool operator>(GmacProtection prot1, GmacProtection prot2)
{
    if ((prot1 == GMAC_PROT_NONE &&
         prot2 != GMAC_PROT_NONE) ||
        (!prot_is_writable(prot1) &&
          prot_is_writable(prot2)
        )
       ) return true;

    return false;
}

static inline
bool operator<=(GmacProtection prot1, GmacProtection prot2)
{
    return !(prot1 > prot2);
}


namespace __impl { namespace dsm {

class GMAC_LOCAL mapping :
    public util::unique<mapping>,
    public util::factory<coherence::block,
                         coherence::block_ptr> {

    friend class util::factory<mapping, mapping_ptr>;
    friend class coherence::block;

protected:
    hal::ptr addr_;
    size_t size_;
    GmacProtection prot_;

    typedef util::factory<coherence::block,
                          coherence::block_ptr> factory_block;

    typedef std::list<coherence::block_ptr> list_block;
    list_block blocks_;

    typedef util::range<list_block::iterator> range_block;
    range_block get_blocks_in_range(size_t offset, size_t count);

    template <bool forward = true>
    void shift_blocks(size_t offset);

#if 0
    template <typename I>
    static mapping_ptr merge_mappings(util::range<I> range, size_t off, size_t count);

    static error dup2(mapping_ptr map1, hal::ptr::offset_type off1,
                            mapping_ptr map2, hal::ptr::offset_type off2, size_t count);

    error dup(hal::ptr::offset_type off1, mapping_ptr map2,
                    hal::ptr::offset_type off2, size_t count);
#endif

    typedef std::pair<mapping_ptr, mapping_ptr> pair_mapping;

    pair_mapping
    split(size_t off, size_t count, error &err);

    //typedef std::pair<list_block::iterator, size_t> pair_block_info;
    class cursor_block {
        list_block::iterator it_;
        size_t offBlock_;
        size_t offLocal_;

    public:
        cursor_block(list_block::iterator it,
                     size_t offBlock,
                     size_t offLocal) :
            it_(it),
            offBlock_(offBlock),
            offLocal_(offLocal)
        {
        }

        coherence::block_ptr get_block()
        {
            return *it_;
        }

        coherence::block_ptr get_block() const
        {
            return *it_;
        }

        size_t get_offset() const
        {
            return offBlock_ + offLocal_;
        }

        size_t get_offset_block() const
        {
            return offBlock_;
        }

        size_t get_offset_local() const
        {
            return offLocal_;
        }

        size_t get_bytes_to_next_block() const;

        size_t advance_block()
        {
            size_t ret = get_bytes_to_next_block();
            ++it_;
            offBlock_ += ret;
            offLocal_ = 0;

            return ret;
        }

        void advance_bytes(size_t bytes)
        {
            ASSERTION(get_bytes_to_next_block() > bytes);
            offLocal_ += bytes;
        }

        void reset(list_block::iterator it, bool keepOffset = true)
        {
            it_ = it;
            if (!keepOffset) {
                offBlock_ = 0;
                offLocal_ = 0;
            }
        }

        list_block::iterator
        get_iterator()
        {
            return it_;
        }
    };

    cursor_block get_first_block(size_t off);

    cursor_block split_block(cursor_block cursor, size_t offset);

    template <bool Pre>
    void block_splitted(coherence::block_ptr blockOld, coherence::block_ptr blockNew)
    {
    	auto it = util::algo::find(blocks_, blockOld);
    	ASSERTION(it != blocks_.end());
    	blocks_.insert(++it, blockNew);
    }

    error prepend(coherence::block_ptr b);
    error append(coherence::block_ptr b);

    error merge(coherence::block_ptr b, coherence::block_ptr bNew);

    static
    error move_block(mapping &dst, mapping &src, coherence::block_ptr b);

    mapping(hal::ptr addr, GmacProtection prot);
    mapping(mapping &&m);

    // Deleted functions
    mapping(const mapping &) = delete;

public:
    virtual ~mapping();

    static const size_t MinAlignment = 4096;

    typedef std::list<mapping_ptr> submappings;

    error acquire(size_t offset, size_t count, GmacProtection prot);
    error release(size_t offset, size_t count);

    static error link(size_t ptr1, mapping_ptr m1,
                      size_t ptr2, mapping_ptr m2, size_t count, int flags);

    unsigned get_nblocks() const;

    typedef util::bounds<size_t> bounds;
    bounds get_bounds() const;

    hal::ptr get_ptr() const;

    GmacProtection get_protection() const;

    error resize(size_t pre, size_t post);

    error append(mapping &&m);

    template <bool Hex>
    void
    print() const;
};

}}

#include "mapping-impl.h"

#endif // GMAC_DSM_MAPPING_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
