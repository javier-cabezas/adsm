#ifndef GMAC_HAL_CUDA_OPERATION_H_
#define GMAC_HAL_CUDA_OPERATION_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "trace/logger.h"

#include "util/unique.h"

namespace __impl { namespace hal {

namespace detail {
    class event;
    class stream;
}
    
namespace cuda {

namespace virt {
    class aspace;
    class buffer;
}

typedef util::shared_ptr<hal::detail::event> event_ptr;
typedef hal::detail::stream hal_stream;

class stream;

class GMAC_LOCAL operation :
    public hal::detail::operation {
    friend class util::factory<operation>;

    typedef hal::detail::operation parent;

private:
    bool synced_;
    CUevent eventStart_;
    CUevent eventEnd_;

    virt::aspace &as_;
    stream *stream_;
    operation(parent::type t, bool async, virt::aspace &as, stream &s);
    virtual ~operation();
    
    hal::error sync();

    void set_barrier(hal::detail::stream &_s);

public:
    void
    reset(parent::type t, bool async, stream &s);

    state get_state();
    template <typename Func, typename... Args>
    auto execute(Func f, Args... args) -> decltype(f(CUstream(), args...));

    stream &get_stream()
    {
        return *stream_;
    }
};

typedef hal::detail::event event;
typedef hal::detail::event_ptr event_ptr;
typedef hal::detail::list_event list_event_detail;
typedef hal::detail::stream hal_stream;

operation *
create_op(operation::type t, bool async, virt::aspace &as, stream &s);

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
