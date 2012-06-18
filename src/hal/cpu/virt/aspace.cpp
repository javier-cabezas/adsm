#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

#include "hal/types.h"

#include "map_file.h"

namespace __impl { namespace hal { namespace cpu { namespace virt {

static map_file Files;

aspace::aspace(hal_aspace::set_processing_unit &compatibleUnits, phys::aspace &pas, hal::error &err) :
    parent(compatibleUnits, pas, err)
{
}

int get_prot_flags(GmacProtection prot)
{
    int ret = 0;

    if (prot & GMAC_PROT_READ) {
        ret |= PROT_READ;
    }
    if (prot & GMAC_PROT_WRITE) {
        ret |= PROT_WRITE;
    }

    return ret;
}

ptr
aspace::map(hal_object &obj, GmacProtection prot, hal::error &err)
{
    return map(obj, prot, 0, err);
}

ptr
aspace::map(hal_object &obj, GmacProtection _prot, size_t offset, hal::error &err)
{
    hal_object::set_view viewsGpu = obj.get_views(phys::hal_processing_unit::PUNIT_TYPE_GPU);
    if (viewsGpu.size() != 0) {
        // Map objects allocated in GPU memory is not supported
        err = hal::error::HAL_ERROR_FEATURE_NOT_SUPPORTED;
        return ptr();
    }

    hal_object::set_view viewsCpu    = obj.get_views(phys::hal_processing_unit::PUNIT_TYPE_CPU);
    hal_object::set_view viewsAspace = obj.get_views(*this);

    if (viewsCpu.size() != viewsAspace.size()) {
        // Mappings across different address spaces is not supported yet
        err = hal::error::HAL_ERROR_FEATURE_NOT_SUPPORTED;
        return ptr();
    }

    int prot = get_prot_flags(_prot);

    if (viewsAspace.size() == 0) {
        /////////////////////
        // Create new mapping
        /////////////////////

        host_ptr cpuAddr = NULL;
        char tmp[FILENAME_MAX];

        // Create new shared memory file
        snprintf(tmp, FILENAME_MAX, "/tmp/gmacXXXXXX");
        int fd = mkstemp(tmp);
        if(fd < 0) {
            return ptr();
        }
        unlink(tmp);

        if (ftruncate(fd, obj.get_size()) < 0) {
            close(fd);
            return ptr();
        }

        if (offset == 0) {
            cpuAddr = host_ptr(::mmap((void *) offset, obj.get_size(), prot, MAP_SHARED, fd, 0));
            TRACE(LOCAL, FMT_ID2" Getting map: %d @ %p - %p", get_print_id2(), prot, cpuAddr, offset + obj.get_size());
        } else {
            cpuAddr = host_ptr(offset);
            if (::mmap(cpuAddr, obj.get_size(), prot, MAP_SHARED | MAP_FIXED, fd, 0) != cpuAddr) {
                close(fd);
                return ptr();
            }
            TRACE(LOCAL, FMT_ID2" Getting fixed map: %d @ %p - %p", get_print_id2(), prot, offset, offset + obj.get_size());
        }

        if (Files.insert(fd, cpuAddr, obj.get_size()) == false) {
            munmap(cpuAddr, obj.get_size());
            close(fd);
            return ptr();
        }

        detail::virt::object_view *view = obj.create_view(*this, hal::ptr::offset_type(cpuAddr), err);

        if (err == hal::error::HAL_SUCCESS) {
            // Add view to the map
            addrsToView_.insert(map_addr_to_view::value_type(hal::ptr::offset_type(cpuAddr) + obj.get_size(), view));
            return ptr(*view);
        } else {
            return ptr();
        }
    } else {
        ////////////////////////
        // Create shadow mapping
        ////////////////////////

        if (viewsAspace.size() > 1) {
            // Supporting up to two mappings for now
            err = hal::error::HAL_ERROR_FEATURE_NOT_SUPPORTED;
            return ptr();
        }
        const detail::virt::object_view *old = *viewsAspace.begin();

        TRACE(GLOBAL, "Getting shadow mapping for %p (%zd bytes)", old->get_offset(), obj.get_size());
        map_file_entry entry = Files.find(host_ptr(old->get_offset()));
        if (entry.fd() == -1) {
            err = hal::error::HAL_ERROR_INVALID_VALUE;
            return ptr();
        }
        off_t off = off_t(host_ptr(old->get_offset()) - entry.address());
        ASSERTION(off == 0, "Offsets are not supported for now");
        int flags = MAP_SHARED;
#if not defined(__APPLE__)
        flags |= MAP_POPULATE;
#endif
        host_ptr ret = host_ptr(::mmap((void *) offset, obj.get_size(), prot, flags, entry.fd(), off));

        detail::virt::object_view *view;
        if (ret != MAP_FAILED) {
            view = obj.create_view(*this, hal::ptr::offset_type(ret), err);
        } else {
            err = hal::error::HAL_ERROR_INVALID_VALUE;
        }

        if (err == hal::error::HAL_SUCCESS) {
            addrsToView_.insert(map_addr_to_view::value_type(hal::ptr::offset_type(ret) + obj.get_size(), view));
            return ptr(*view);
        } else {
            return ptr();
        }
    }
}

detail::code::repository_view *
aspace::map(const detail::code::repository &repo, hal::error &err)
{
    NOT_IMPLEMENTED();
    return NULL;
}

hal::error
aspace::unmap(ptr p)
{
    detail::virt::object_view &view = p.get_view();
    detail::virt::object &obj = view.get_object();

    host_ptr ptr = host_ptr(view.get_offset()) + p.get_offset();

    obj.destroy_view(view);
    if (obj.get_views().size() == 0) {
        map_file_entry entry = Files.find(host_ptr(ptr));
        if (Files.remove(ptr) == false) {
            return hal::error::HAL_ERROR_INVALID_VALUE;
        }
        close(entry.fd());
    }
    ::munmap(ptr, obj.get_size());

    return hal::error::HAL_SUCCESS;
}

hal::error
aspace::unmap(detail::code::repository_view &view)
{
    NOT_IMPLEMENTED();
    return hal::error::HAL_SUCCESS;
}

hal::error
aspace::protect(hal::ptr _ptr, size_t count, GmacProtection prot)
{
    hal::error ret = hal::error::HAL_SUCCESS;
    int p = PROT_NONE;
    switch (prot) {
    case GmacProtection::GMAC_PROT_READ:
        p = PROT_READ;
        break;
    case GmacProtection::GMAC_PROT_WRITE:
        p = PROT_WRITE;
        break;
    case GmacProtection::GMAC_PROT_READWRITE:
        p = PROT_READ | PROT_WRITE;
        break;
    case GmacProtection::GMAC_PROT_NONE:
        break;
    };
    void *ptr = get_host_ptr(_ptr);
    int res = ::mprotect(ptr, count, p);
    if (res != 0) {
        ret = hal::error::HAL_ERROR_INVALID_VALUE;
    }

    return ret;
}

bool
aspace::has_direct_copy(hal::const_ptr ptr1, hal::const_ptr ptr2)
{
    // TODO: refine this logic
    return &ptr1.get_view().get_vaspace() == &ptr2.get_view().get_vaspace();
}

hal_event_ptr
aspace::copy(hal::ptr dst, hal::const_ptr src, size_t count, list_event_detail *dependencies, hal::error &err)
{
    if (is_host_ptr(dst) &&
        is_host_ptr(src)) {
        ::memcpy(get_host_ptr(dst), get_host_ptr(src), count);
    } else {
        FATAL("Unhandled case");
    }

    return hal_event_ptr();
}

hal_event_ptr
aspace::copy_async(hal::ptr dst, hal::const_ptr src, size_t count, list_event_detail *dependencies, hal::error &err)
{
    NOT_IMPLEMENTED();

    return hal_event_ptr();
}

hal_event_ptr
aspace::copy(hal::ptr dst, device_input &input, size_t count, list_event_detail *dependencies, hal::error &err)
{
    NOT_IMPLEMENTED();

    return hal_event_ptr();
}

hal_event_ptr
aspace::copy(device_output &output, hal::const_ptr src, size_t count, list_event_detail *dependencies, hal::error &err)
{
    NOT_IMPLEMENTED();

    return hal_event_ptr();
}

hal_event_ptr
aspace::memset(hal::ptr dst, int c, size_t count, list_event_detail *dependencies, hal::error &err)
{
    NOT_IMPLEMENTED();

    return hal_event_ptr();
}

hal_event_ptr
aspace::copy_async(hal::ptr dst, device_input &input, size_t count, list_event_detail *dependencies, hal::error &err)
{
    NOT_IMPLEMENTED();

    return hal_event_ptr();
}

hal_event_ptr
aspace::copy_async(device_output &output, hal::const_ptr src, size_t count, list_event_detail *dependencies, hal::error &err)
{
    NOT_IMPLEMENTED();

    return hal_event_ptr();
}

hal_event_ptr
aspace::memset_async(hal::ptr dst, int c, size_t count, list_event_detail *dependencies, hal::error &err)
{
    NOT_IMPLEMENTED();

    return hal_event_ptr();
}

hal_context *
aspace::create_context(hal::error &err)
{
    FATAL("Context creation not implemented on CPU");
    return nullptr;
}

void
aspace::destroy_context(hal_context &ctx)
{
    delete &ctx;
}

}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
