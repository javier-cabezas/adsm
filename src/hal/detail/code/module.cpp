#include "hal/detail/types.h"

namespace __impl { namespace hal { namespace detail { namespace code {

hal::error
repository::load_from_file(const std::string &path,
                           const std::string &flags,
                           const util::taggeable<>::set_tag &tags)
{
    for (auto desc : descriptorsFile_) {
        if (desc.get_path() == path) {
            // TODO: update error code
            return HAL_ERROR_INVALID_VALUE;
        }
    }

    descriptor_file desc(path, flags);
    desc.add_tags(tags);

    descriptorsFile_.push_back(desc);
    return HAL_SUCCESS;
}

hal::error
repository::load_from_mem(const char *ptr,
                          size_t size,
                          const std::string &flags,
                          const util::taggeable<>::set_tag &tags)
{
    for (auto desc : descriptorsBuffer_) {
        if (desc.get_ptr() == ptr) {
            // TODO: update error code
            return HAL_ERROR_INVALID_VALUE;
        }
    }

    descriptor_buffer desc(ptr, size, flags);
    desc.add_tags(tags);

    descriptorsBuffer_.push_back(desc);
    return HAL_SUCCESS;
}

hal::error
repository::load_from_handle(const void *handle,
                             const std::string &flags,
                             const util::taggeable<>::set_tag &tags)
{
    for (auto desc : descriptorsHandle_) {
        if (desc.get_handle() == handle) {
            // TODO: update error code
            return HAL_ERROR_INVALID_VALUE;
        }
    }

    descriptor_handle desc(handle, flags);
    desc.add_tags(tags);

    descriptorsHandle_.push_back(desc);
    return HAL_SUCCESS;
}

}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
