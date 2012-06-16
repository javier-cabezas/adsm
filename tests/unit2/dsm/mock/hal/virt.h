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
    };
}}}

#endif
