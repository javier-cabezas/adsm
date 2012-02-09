#ifndef GMAC_HAL_OPENCL_TYPES_H_
#define GMAC_HAL_OPENCL_TYPES_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "hal/types-detail.h"

#include "util/stl/locked_map.h"

#include "ptr.h"

namespace __impl { namespace hal { namespace opencl {

class device;
class platform;

class coherence_domain;
class aspace;
class stream_t;
class kernel_t;
class texture_t;
class variable_t;
class code_repository;
class _event_t;
class event_ptr;
class event_list;
class buffer_t;

typedef __impl::hal::_ptr_t<host_ptr,
                            _opencl_ptr_t,
                            aspace> ptr_t;

typedef __impl::hal::_ptr_t<host_const_ptr,
                            _opencl_ptr_t,
                            aspace> ptr_const_t;

typedef hal::detail::backend_traits<cl_context,
                                    cl_command_queue,
                                    cl_event,
                                    cl_kernel,
                                    size_t *> backend_traits;

typedef hal::detail::implementation_traits<coherence_domain,
                                           aspace,
                                           stream_t,
                                           kernel_t,
                                           texture_t,
                                           variable_t,
                                           code_repository,
                                           _event_t,
                                           event_ptr,
                                           event_list,
                                           buffer_t,
                                           ptr_t,
                                           ptr_const_t> implementation_traits;

typedef util::stl::locked_map<platform *, code_repository> map_platform_repository;

extern map_platform_repository Modules_;

gmacError_t error(cl_int err);
gmacError_t compile_embedded_code(std::list<device *> devices);
gmacError_t compile_code(platform &plat, const std::string &code, const std::string &flags);
gmacError_t compile_binary(platform &plat, const std::string &code, const std::string &flags);

}}}

#include "event.h"
#include "aspace.h"
#include "kernel.h"
#include "stream.h"

#include "aspace-impl.h"
#include "stream-impl.h"
#include "kernel-impl.h"
#include "event-impl.h"

#include "module.h"

#endif /* GMAC_HAL_OPENCL_TYPES_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
