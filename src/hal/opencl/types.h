#ifndef GMAC_HAL_OPENCL_TYPES_H_
#define GMAC_HAL_OPENCL_TYPES_H_

#include <CL/cl.h>

#include "hal/types-detail.h"

#include "ptr.h"

namespace __impl { namespace hal { namespace opencl {

class device;
class platform;

class coherence_domain;
class context_t;
class stream_t;
class kernel_t;
class texture_t;
class variable_t;
class code_repository;
class _event_t;
class event_t;
class event_list;
class buffer_t;

typedef __impl::hal::_ptr_t<_opencl_ptr_t,
                            context_t> ptr_t;

typedef hal::detail::backend_traits<cl_context,
                                    cl_command_queue,
                                    cl_event,
                                    cl_kernel,
                                    size_t *> backend_traits;

typedef hal::detail::implementation_traits<coherence_domain,
                                           context_t,
                                           stream_t,
                                           kernel_t,
                                           texture_t,
                                           variable_t,
                                           code_repository,
                                           event_t,
                                           event_list,
                                           buffer_t,
                                           ptr_t> implementation_traits;

gmacError_t error(cl_int err);
gmacError_t compile_embedded_code(std::list<device *> devices);
gmacError_t compile_code(platform &plat, const std::string &code, const std::string &flags);
gmacError_t compile_binary(platform &plat, const std::string &code, const std::string &flags);

}}}

#include "event.h"
#include "context.h"
#include "kernel.h"
#include "stream.h"

#include "context-impl.h"
#include "stream-impl.h"
#include "kernel-impl.h"
#include "event-impl.h"

#include "module.h"

#endif /* GMAC_HAL_OPENCL_TYPES_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
