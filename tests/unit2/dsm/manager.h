#include "common.h"

#include "mock/hal/types.h"

#include "gtest/gtest.h"

#include "dsm/manager.h"

#include "util/misc.h"

#include "unit2/dsm/mapping.h"

using I_UTIL::range;
using I_HAL::const_ptr;
using I_HAL::ptr;

using I_DSM::coherence::block;
using I_DSM::coherence::block_ptr;

class manager_mapping_test : public testing::Test {
public:

protected:
	static void SetUpTestCase();
	static void TearDownTestCase();
};

class manager;
typedef manager *manager_ptr;

class GMAC_LOCAL manager :
    public I_DSM::manager {

    typedef I_DSM::manager parent;

public:
    // Forward types
    typedef parent::range_mapping range_mapping;
    typedef parent::map_mapping_group map_mapping_group;

    // Forward protected functions
    static void aspace_created(manager *m, I_HAL::virt::aspace &aspace);
    static void aspace_destroyed(manager *m, I_HAL::virt::aspace &aspace);

    template <bool IsOpen>
    range_mapping
    get_mappings_in_range(map_mapping_group &mappings, ptr begin, size_t count)
    {
        return parent::get_mappings_in_range<IsOpen>(mappings, begin, count);
    }

    map_mapping_group &
    get_aspace_mappings(I_HAL::virt::aspace &aspace)
    {
        return parent::get_aspace_mappings(aspace);
    }

    I_DSM::mapping_ptr
    merge_mappings(range_mapping &range)
    {
        return parent::merge_mappings(range);
    }

    I_DSM::error
    replace_mappings(map_mapping_group &mappings, range_mapping &range, I_DSM::mapping_ptr mNew)
    {
        return parent::replace_mappings(mappings, range, mNew);
    }

    virtual
    ~manager()
    {
    }

    /**
     * Default constructor
     */
    manager() :
        parent()
    {
    }

#if 0
    /**
     * Map the given host memory pointer to the accelerator memory. If the given
     * pointer is NULL, host memory is alllocated too.
     * \param mode Execution mode where to allocate memory
     * \param addr Memory address to be mapped or NULL if host memory is requested
     * too
     * \param size Size (in bytes) of shared memory to be mapped 
     * \param flags 
     * \param err Reference to store the error code for the operation
     * \return Address that identifies the allocated memory
     */
    gmacError_t link(I_HAL::ptr ptr1,
                     I_HAL::ptr ptr2, size_t count, int flags);
#endif

    I_DSM::error unlink(ptr mapping, size_t count);

    I_DSM::error acquire(ptr mapping, size_t count, int flags);
    I_DSM::error release(ptr mapping, size_t count);

    I_DSM::error sync(ptr mapping, size_t count);

    I_DSM::error memcpy(ptr dst, ptr src, size_t count);
    I_DSM::error memset(ptr p, int c, size_t count);

    I_DSM::error from_io_device(ptr addr, I_HAL::device_input &input, size_t count);
    I_DSM::error to_io_device(I_HAL::device_output &output, const_ptr addr, size_t count);

    I_DSM::error flush_dirty(I_DSM::address_space_ptr aspace);

    //////////////////////
    // Helper functions //
    //////////////////////
    bool
    helper_insert(I_HAL::virt::aspace &as, mapping_ptr m)
    {
        I_DSM::error ret;

        parent::map_mapping_group &group = parent::get_aspace_mappings(as);
        ret = parent::insert_mapping(group, m);

        return ret == I_DSM::error::DSM_SUCCESS;
    }

    bool
    helper_delete_mappings(I_HAL::virt::aspace &as)
    {
        parent::map_mapping_group &group = parent::get_aspace_mappings(as);

        return parent::delete_mappings(group) == I_DSM::error::DSM_SUCCESS;
    }

    parent::map_mapping &
    helper_get_mappings(I_HAL::virt::aspace &as, I_HAL::virt::object_view &view)
    {
        parent::map_mapping_group &group = parent::get_aspace_mappings(as);

        parent::map_mapping_group::iterator it = group.find(&view);
        ASSERTION(it != group.end());

        return it->second;
    }

    I_DSM::mapping_ptr
    helper_get_mapping(ptr p)
    {
        if (!bool(p)) return nullptr;

        parent::map_mapping_group &group = parent::get_aspace_mappings(p.get_view().get_vaspace());

        parent::range_mapping range = parent::get_mappings_in_range<false>(group, p, 1);
        if (range.is_empty()) {
            return nullptr;
        }

        return *range.begin();
    }
};
