#ifndef GMAC_HAL_PHYS_PROCESSING_UNIT_H_
#define GMAC_HAL_PHYS_PROCESSING_UNIT_H_

#include <set>

#include "config/common.h"
#include "util/unique.h"

#include "memory.h"

namespace __impl { namespace hal {
    
class list_platform;
class platform;

list_platform get_platforms();

namespace detail {
    
namespace virt {
    class aspace;
}

class stream;

namespace phys {

class aspace;
class coherence_domain;
class memory;
class platform;

class GMAC_LOCAL processing_unit :
    public util::unique<processing_unit> {
    friend list_platform hal::get_platforms();
public:
    struct memory_connection {
        memory *mem;
        bool direct;
        unsigned long latency;

        memory_connection(memory &_mem, bool _direct, unsigned long _latency) :
            mem(&_mem),
            direct(_direct),
            latency(_latency)
        {
        }

        bool operator<(const memory_connection &connection) const
        {
            return mem < connection.mem;
        }

        memory_connection &operator=(const memory_connection &connection) = default;
    };

    typedef std::set<memory_connection> set_memory_connection;
    typedef std::set<aspace *>          set_aspace;

    enum type {
        PUNIT_TYPE_CPU = 0,
        PUNIT_TYPE_GPU = 1
    };
protected:
    platform &platform_;
    bool integrated_;
    type type_;
    aspace &as_;

    set_memory_connection connections_;
    // set_aspace aspaces_;

    processing_unit(platform &platform, type t, aspace &as);
public:
    virtual stream *create_stream(virt::aspace &as) = 0;
    virtual gmacError_t destroy_stream(stream &stream) = 0;

    void add_memory_connection(const memory_connection &connection);

    //platform &get_platform();
    const platform &get_platform() const;

    aspace &get_paspace();
    const aspace &get_paspace() const;

    bool is_integrated() const;
    type get_type() const;

    bool has_access(const memory &mem, memory_connection &connection);

    virtual size_t get_total_memory() const = 0;
    virtual size_t get_free_memory() const = 0;

    virtual gmacError_t get_info(GmacDeviceInfo &info) = 0;
};

}}}}

#endif /* GMAC_HAL_PHYS_PROCESSING_UNIT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
