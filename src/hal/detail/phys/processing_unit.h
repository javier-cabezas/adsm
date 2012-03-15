#ifndef GMAC_HAL_PHYS_PROCESSING_UNIT_H_
#define GMAC_HAL_PHYS_PROCESSING_UNIT_H_

#include <set>

#include "config/common.h"
#include "util/unique.h"

#include "memory.h"

namespace __impl { namespace hal {
    
namespace detail {
    
namespace virt {
    class aspace;
}

class stream;

namespace phys {

class aspace;
typedef util::shared_ptr<aspace> aspace_ptr;

class coherence_domain;
class memory;
class platform;

class GMAC_LOCAL processing_unit :
    public util::unique<processing_unit> {
public:
    struct memory_connection {
        memory_ptr mem;
        unsigned long latency;

        memory_connection(memory_ptr _mem, unsigned long _latency) :
            mem(_mem),
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
    typedef std::set<aspace_ptr>        set_aspace;

    enum type {
        PUNIT_TYPE_CPU = 0,
        PUNIT_TYPE_GPU = 1
    };
protected:
    platform &platform_;
    bool integrated_;
    type type_;

    set_memory_connection memories_;
    set_aspace aspaces_;

    processing_unit(type t, platform &platform,
                    set_memory_connection &memories,
                    set_aspace &aspaces);
public:
    virtual stream *create_stream(virt::aspace &as) = 0;
    virtual gmacError_t destroy_stream(stream &stream) = 0;

    platform &get_platform();
    const platform &get_platform() const;

    set_aspace &get_paspaces();
    const set_aspace &get_paspaces() const;

    bool is_integrated() const;
    type get_type() const;

    bool has_access(memory_ptr mem, memory_connection &connection);

    virtual size_t get_total_memory() const = 0;
    virtual size_t get_free_memory() const = 0;

    virtual gmacError_t get_info(GmacDeviceInfo &info) = 0;
};

}}}}

#endif /* GMAC_HAL_PHYS_PROCESSING_UNIT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
