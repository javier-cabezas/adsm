#ifndef GMAC_HAL_DETAIL_VIRT_ASPACE_H_
#define GMAC_HAL_DETAIL_VIRT_ASPACE_H_

#include <list>
#include <queue>

#include "trace/logger.h"

#include "util/attribute.h"
#include "util/lock.h"
#include "util/locked_counter.h"
#include "util/trigger.h"

#include "hal/error.h"
#include "hal/detail/ptr.h"

namespace __impl { namespace hal { namespace detail {
    
namespace code {
    class kernel;
    class repository;
    class repository_mapping;
}

class event;
class stream;
class list_event;
typedef util::shared_ptr<event> event_ptr;

namespace virt {

class aspace;
class context;
class object;

class GMAC_LOCAL handler_sigsegv {
    static void fun_null() {}
    
public:
    typedef std::function<bool (hal::ptr, bool)> function;
    typedef std::function<void (void)>   fn_;

    typedef std::pair<fn_, fn_> pair_fn;

    function handler_;
    pair_fn fns_;

private:

public:
    handler_sigsegv(function handler,
                    pair_fn ctx = pair_fn(fun_null, fun_null)) :
        handler_(handler),
        fns_(ctx)
    {
    }

    void exec_pre()
    {
        fns_.first();
    }

    void exec_post()
    {
        fns_.second();
    }

    bool exec(hal::ptr p, bool isWrite)
    {
        return handler_(p, isWrite);
    }
};

class GMAC_LOCAL aspace :
    public util::unique<aspace, GmacAddressSpaceId>,
    public util::attributes<aspace>,
    public util::observable<aspace, util::event::construct>,
    public util::observable<aspace, util::event::destruct> {
    friend class util::observable<aspace, util::event::destruct>;

public:
    typedef std::set<phys::processing_unit *> set_processing_unit;
    typedef util::observable<aspace, util::event::construct> observe_class_construct;
    typedef util::observable<aspace, util::event::destruct>  observe_class_destruct;

protected:
    set_processing_unit pUnits_;

    phys::aspace &pas_;

    aspace(set_processing_unit &compatibleUnits, phys::aspace &pas, hal::error &err);
    virtual ~aspace();

public:
    const set_processing_unit &get_processing_units() const;

    phys::aspace &get_paspace();
    const phys::aspace &get_paspace() const;

    virtual ptr map(virt::object &obj, GmacProtection prot, hal::error &err) = 0;
    virtual ptr map(virt::object &obj, GmacProtection prot, size_t offset, hal::error &err) = 0;

    virtual code::repository_view *map(const code::repository &repo, hal::error &err) = 0;

    virtual hal::error unmap(ptr p) = 0;
    virtual hal::error unmap(code::repository_view &view) = 0;

    virtual hal::error protect(hal::ptr ptr, size_t count, GmacProtection prot) = 0;

    virtual hal::error      handler_sigsegv_push(handler_sigsegv &handler) = 0;
    virtual handler_sigsegv handler_sigsegv_pop(hal::error &err) = 0;

    virtual bool has_direct_copy(hal::const_ptr ptr1, hal::const_ptr ptr2) = 0;

    virtual event_ptr copy(hal::ptr dst, hal::const_ptr src, size_t count, list_event *dependencies, hal::error &err) = 0;
    virtual event_ptr copy_async(hal::ptr dst, hal::const_ptr src, size_t count, list_event *dependencies, hal::error &err) = 0;

    virtual event_ptr copy(hal::ptr dst, device_input &input, size_t count, list_event *dependencies, hal::error &err) = 0;
    virtual event_ptr copy(device_output &output, hal::const_ptr src, size_t count, list_event *dependencies, hal::error &err) = 0;
    virtual event_ptr memset(hal::ptr dst, int c, size_t count, list_event *dependencies, hal::error &err) = 0;
    virtual event_ptr copy_async(hal::ptr dst, device_input &input, size_t count, list_event *dependencies, hal::error &err) = 0;
    virtual event_ptr copy_async(device_output &output, hal::const_ptr src, size_t count, list_event *dependencies, hal::error &err) = 0;
    virtual event_ptr memset_async(hal::ptr dst, int c, size_t count, list_event *dependencies, hal::error &err) = 0;

    virtual context *create_context(hal::error &err) = 0;
};

}

}}}

#endif /* ASPACE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
