#ifndef GMAC_TEST_UNIT_MOCK_HAL_VIRT_H_
#define GMAC_TEST_UNIT_MOCK_HAL_VIRT_H_

#include <stddef.h>

#include "include/gmac/visibility.h"
#include "util/attribute.h"
#include "util/trigger.h"
#include "util/unique.h"

#include "hal/error.h"

#include "ptr.h"

namespace __impl { namespace hal { namespace virt {
    class aspace;

    class GMAC_LOCAL object_view {
        aspace &as_;
        ptrdiff_t ptr_;
    public:
        object_view(aspace &as, ptrdiff_t ptr) :
            as_(as),
            ptr_(ptr)
        {
        }

        aspace &get_vaspace()
        {
            return as_;
        }

        const aspace &get_vaspace() const
        {
            return as_;
        }
    };

    class GMAC_LOCAL handler_sigsegv {
        static void fun_null() {}
    public:
        typedef std::function<bool (hal::ptr, bool)> function;
        typedef std::function<void (void)>   fn_;

        typedef std::pair<fn_, fn_> pair_fn;

    public:
        handler_sigsegv(function handler,
                        pair_fn ctx = pair_fn(fun_null, fun_null))
        {
        }

        void exec_pre()
        {
        }

        void exec_post()                                                                                                   
        {                                                                                                                  
        } 
    };

    class GMAC_LOCAL aspace :
        public util::unique<aspace, GmacAddressSpaceId>,
        public util::attributes<aspace>,
        public util::observable<aspace, util::event::construct>,
        public util::observable<aspace, util::event::destruct>{
        friend class util::observable<aspace, util::event::construct>;
        friend class util::observable<aspace, util::event::destruct>;
    public:
        typedef util::observable<aspace, util::event::construct> observe_construct;
        typedef util::observable<aspace, util::event::destruct>  observe_destruct;

    private:
        std::list<handler_sigsegv> handlers_;

        aspace()
        {
            TRACE(LOCAL, FMT_ID2" created", get_print_id2());
        }

        ~aspace()
        {
        }

    public:
        static aspace *create()
        {
            return observe_construct::create<aspace>();
        }

        static void destroy(aspace &as)
        {
            observe_destruct::destroy(as);
        }

        error protect(hal::ptr p, size_t count, GmacProtection prot)
        {
            CHECK(&p.get_view().get_vaspace() == this, error::HAL_ERROR_INVALID_PTR);

            return error::HAL_SUCCESS;
        }

        error handler_sigsegv_push(handler_sigsegv &handler)
        {
            handlers_.push_back(handler);
            return error::HAL_SUCCESS;
        }

        handler_sigsegv handler_sigsegv_pop(error &err)
        {
            return handlers_.back();
        }
    };
}}}

#endif
