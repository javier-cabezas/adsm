#ifndef GMAC_HAL_DETAIL_CODE_REPOSITORY_H_
#define GMAC_HAL_DETAIL_CODE_REPOSITORY_H_

#include <string>

namespace __impl { namespace hal { namespace detail { namespace code {

class kernel;

class GMAC_LOCAL repository
{
public:
    virtual gmacError_t load_from_file(const std::string &path,
                                       const std::string &flags) = 0;

    virtual gmacError_t load_from_mem(const char *ptr,
                                      size_t size,
                                      const std::string &flags) = 0;

    virtual gmacError_t load_from_handle(const char *ptr,
                                         const std::string &flags) = 0;
};


class GMAC_LOCAL repository_mapping
{
public:
    virtual kernel *get_kernel(gmac_kernel_id_t key) = 0;
    virtual kernel *get_kernel(const std::string &name) = 0;
};
    
}}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
