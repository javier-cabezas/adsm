#include "hal/detail/types.h"

namespace __impl { namespace hal { namespace detail { namespace code {

gmacError_t
repository::load_from_file(const std::string &path,
                           const std::string &flags)
{
    for (auto desc : descriptorsFile_) {
        if (desc.get_path() == path) {
            // TODO: update error code
            return gmacErrorInvalidValue;
        }
    }

    descriptorsFile_.push_back(descriptor_file(path, flags));
    return gmacSuccess;
}

gmacError_t
repository::load_from_mem(const char *ptr,
                          size_t size,
                          const std::string &flags)
{
    for (auto desc : descriptorsBuffer_) {
        if (desc.get_ptr() == ptr) {
            // TODO: update error code
            return gmacErrorInvalidValue;
        }
    }

    descriptorsBuffer_.push_back(descriptor_buffer(ptr, size, flags));
    return gmacSuccess;
}

gmacError_t
repository::load_from_handle(const void *handle,
                             const std::string &flags)
{
    for (auto desc : descriptorsHandle_) {
        if (desc.get_handle() == handle) {
            // TODO: update error code
            return gmacErrorInvalidValue;
        }
    }

    descriptorsHandle_.push_back(descriptor_handle(handle, flags));
    return gmacSuccess;
}

}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
