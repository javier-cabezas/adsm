#include "types.h"

namespace __impl { namespace hal { namespace opencl {

kernel_t::arg_list::map_global_subbuffer kernel_t::arg_list::mapSubBuffer_;

cl_mem
kernel_t::arg_list::get_subbuffer(cl_context context, hostptr_t ptr, ptr_t accPtr, size_t size)
{
    cl_mem ret;

    cache_subbuffer::const_iterator itCacheMap = cacheSubBuffer_.find(ptr);
    if (itCacheMap == cacheSubBuffer_.end()) {
        // find always returns a value
        map_global_subbuffer::iterator itGlobalMap = mapSubBuffer_.find_context(context);

        map_subbuffer &mapMode = itGlobalMap->second;
        map_subbuffer::iterator itModeMap = mapMode.find(ptr);

        if (itModeMap == mapMode.end()) {
            int err;
            cl_buffer_region region;
            region.origin = accPtr.get_offset();
            region.size   = size - accPtr.get_offset();
            ret = clCreateSubBuffer(accPtr.get_device_addr(), CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
            ASSERTION(err == CL_SUCCESS);

            mapMode.insert(map_subbuffer::value_type(ptr, cl_mem_ref(ret, 0)));
            itModeMap = mapMode.find(ptr);
        }

        ret = itModeMap->second.first;
        itModeMap->second.second++;
        cacheSubBuffer_.insert(cache_subbuffer::value_type(ptr, cache_entry(context, itModeMap)));
    } else {
        // Cache-entry -> cache_entry -> iterator -> pair
        ret = itCacheMap->second.second->second.first;
    }

    return ret;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
