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

#ifndef GMAC_DSM_MANAGER_H_
#define GMAC_DSM_MANAGER_H_

#include "config/common.h"
#include "include/gmac/types.h"
#include "util/iterator.h"
#include "util/singleton.h"
#include "util/stl/locked_map.h"
#include "util/trigger.h"

#include "mapping.h"

namespace __impl {

namespace dsm {

//! DSM Manager Interface

//! DSM Managers orchestate the data transfers between host and accelerator memories
class GMAC_LOCAL manager :
    public util::singleton<dsm::manager>,
    public util::unique<dsm::manager>,
    public util::observer_class<hal::virt::aspace, util::event::construct>,
    public util::observer_class<hal::virt::aspace, util::event::destruct>,
    public util::factory<mapping, mapping_ptr> {

    friend class util::singleton<dsm::manager>;

    typedef util::singleton<dsm::manager> parent_singleton;

    typedef util::observer_class<hal::virt::aspace, util::event::construct> observer_construct;
    typedef util::observer_class<hal::virt::aspace, util::event::destruct>  observer_destruct;

protected:
    typedef util::factory<mapping, mapping_ptr> factory_mapping;

    typedef std::map<size_t, mapping_ptr> map_mapping;
    typedef map_mapping *map_mapping_ptr;

    typedef std::map<hal::virt::object_view *, map_mapping> map_mapping_group;

    friend bool mapping_fits(map_mapping &, mapping_ptr);

    const hal::virt::aspace::attribute_id AttributeMappings_;
    const hal::virt::aspace::attribute_id AttributeProtection_;

    void event_handler(hal::virt::aspace &aspace, const util::event::construct &);
    void event_handler(hal::virt::aspace &aspace, const util::event::destruct &);

    map_mapping_group &get_aspace_mappings(hal::virt::aspace &as);

    typedef util::range<util::iterator_wrap_associative_second<map_mapping::iterator> > range_mapping;
    template <bool GetAdjacent>
    static
    range_mapping get_mappings_in_range(map_mapping_group &mappings, hal::ptr begin, size_t count);

    template <bool All>
    static
    bool range_has_protection(const range_mapping &range, GmacProtection prot);

    error insert_mapping(map_mapping_group &mappings, mapping_ptr m);

    mapping_ptr merge_mappings(range_mapping &range);
    
    error replace_mappings(map_mapping_group &mappings, range_mapping &range, mapping_ptr mNew);

    error delete_mappings(map_mapping_group &mappings);

    /**
     * Default destructor
     */
    virtual ~manager();

public:
    /**
     * Default constructor
     */
    manager();

    error link(hal::ptr dst, hal::ptr src, size_t count, GmacProtection protDst, GmacProtection protSrc, int flags = mapping_flags::MAP_DEFAULT);
    error unlink(hal::ptr mapping, size_t count);

    template <bool Hex, bool PrintBlocks>
    static
    void range_print(const range_mapping &range);

    template <bool Hex, bool PrintBlocks>
    void print_all_mappings(hal::virt::aspace &as);

    error acquire(hal::ptr mapping, size_t count, GmacProtection prot);
    error release(hal::ptr mapping, size_t count);

    error sync(hal::ptr mapping, size_t count);

    error memcpy(hal::ptr dst, hal::ptr src, size_t count);
    error memset(hal::ptr ptr, int c, size_t count);

    error from_io_device(hal::ptr addr, hal::device_input &input, size_t count);
    error to_io_device(hal::device_output &output, hal::const_ptr addr, size_t count);

    error flush_dirty(address_space_ptr aspace);

    static
    bool handle_fault(hal::ptr p, bool isWrite);

    error use_memory_protection(hal::virt::aspace &as);

    void destroy_singleton()
    {
        parent_singleton::destroy();
    }
};

}}

#include "manager-impl.h"

#endif
