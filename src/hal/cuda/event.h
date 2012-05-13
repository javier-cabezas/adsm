#ifndef GMAC_HAL_CUDA_EVENT_H_
#define GMAC_HAL_CUDA_EVENT_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "trace/logger.h"

#include "util/unique.h"

namespace __impl { namespace hal {

namespace detail {
    class _event;
    class stream;
}
    
namespace cuda {

namespace virt {
    class aspace;
    class buffer;
}

typedef util::shared_ptr<hal::detail::_event> hal_event_ptr;
typedef hal::detail::stream hal_stream;

class stream;

#if 0
typedef std::function<CUresult(void *, const void *, size_t, CUstream)> memcpy_op;
typedef std::function<CUresult(void *, int, size_t, CUstream)>          memset_op;
#endif

class GMAC_LOCAL operation :
    public hal::detail::operation {
    friend class _event_t;

    typedef hal::detail::operation parent;

public:
    typedef std::function<CUresult(CUstream)> func_op;

private:
    bool synced_;
    CUevent eventStart_;
    CUevent eventEnd_;

    stream &stream_;
    operation(parent::type t, bool async, stream &s);

    func_op::result_type execute(func_op f);
    hal::error sync();

    void set_barrier(hal::detail::stream &_s);

public:
    state get_state();
};

#if 0
class GMAC_LOCAL _event_common_t :
    public hal::detail::_event {
    friend class code::kernel;
    friend class virt::aspace;
    friend class virt::buffer;
    friend class device;
    friend class list_event;

    typedef hal::detail::_event parent;

protected:

    // Not instantiable
    _event_common_t(bool async, parent::type t, virt::aspace &context);

    template <typename R>
    R add_operation(hal_event_ptr ptr, stream &stream, std::function<R()> f);

#if 0
    typename memcpy_op::result_type add_operation(hal_event_ptr ptr, stream &stream, memcpy_op op, void *dst, const void *src, size_t count);
#endif

    //stream &get_stream();
    hal::error sync_no_exec();
    hal::error sync();

    state get_state();

public:
    void set_barrier(virt::aspace &as, CUstream stream);
};
#endif

class GMAC_LOCAL _event_t :
    public hal::detail::_event {

    friend class virt::aspace;
    friend class code::kernel;

    typedef hal::detail::_event parent;

    virt::aspace &as_;

protected:
    virtual void reset(bool async, type t);

    _event_t(bool async, parent::type t, virt::aspace &as);

    typename operation::func_op::result_type
    add_operation(hal_event_ptr ptr, stream &stream, operation::func_op f, operation::type t, bool async);
public:
    void set_synced();
    hal::error sync();

    state get_state();
    void set_barrier();

    virt::aspace &get_vaspace();
};

class GMAC_LOCAL event_deleter {
public:
    void operator()(_event_t *ev);
};

static
event_ptr
create_event(bool async, _event_t::type t, virt::aspace &as);

typedef hal::detail::event_ptr hal_event_ptr;
typedef hal::detail::list_event list_event_detail;
typedef hal::detail::stream hal_stream;

class GMAC_LOCAL list_event :
    public list_event_detail {
    typedef list_event_detail parent;

    friend class virt::aspace;
    friend class stream;
    friend class code::kernel;

    //void set_barrier(hal_stream &stream);
public:
    hal::error sync();
    void set_synced();
};

}}}

#endif /* GMAC_HAL_CUDA_EVENT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
