#include "gtest/gtest.h"

#include "dsm/manager.h"
#include "hal/types.h"
#include "util/misc.h"

#include "unit2/dsm/mapping.h"

using __impl::util::range;
using __impl::hal::const_ptr;
using __impl::hal::ptr;

using __impl::dsm::coherence::block;
using __impl::dsm::coherence::block_ptr;

typedef __impl::dsm::error error_dsm;

class manager_mapping_test : public testing::Test {
public:

protected:
	static void SetUpTestCase();
	static void TearDownTestCase();
};

class manager;
typedef manager *manager_ptr;

class GMAC_LOCAL manager :
    public __impl::dsm::manager {

    typedef __impl::dsm::manager parent;

public:
    // Forward types
    typedef parent::range_mapping range_mapping;
    typedef parent::map_mapping_group map_mapping_group;

    // Forward protected functions
    static void aspace_created(manager *m, __impl::hal::aspace &aspace);
    static void aspace_destroyed(manager *m, __impl::hal::aspace &aspace);

    template <bool IsOpen>
    range_mapping
    get_mappings_in_range(map_mapping_group &mappings, ptr begin, size_t count)
    {
        return parent::get_mappings_in_range<IsOpen>(mappings, begin, count);
    }

    map_mapping_group &
    get_aspace_mappings(__impl::hal::aspace &aspace)
    {
        return parent::get_aspace_mappings(aspace);
    }

    __impl::dsm::mapping_ptr
    merge_mappings(range_mapping &range)
    {
        return parent::merge_mappings(range);
    }

    error_dsm
    replace_mappings(map_mapping_group &mappings, range_mapping &range, __impl::dsm::mapping_ptr mNew)
    {
        return parent::replace_mappings(mappings, range, mNew);
    }


    virtual ~manager()
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
    gmacError_t link(__impl::hal::ptr ptr1,
                     __impl::hal::ptr ptr2, size_t count, int flags);
#endif

    error_dsm unlink(ptr mapping, size_t count);

    error_dsm acquire(ptr mapping, size_t count, int flags);
    error_dsm release(ptr mapping, size_t count);

    error_dsm sync(ptr mapping, size_t count);

    error_dsm memcpy(ptr dst, ptr src, size_t count);
    error_dsm memset(ptr p, int c, size_t count);

    error_dsm from_io_device(ptr addr, __impl::hal::device_input &input, size_t count);
    error_dsm to_io_device(__impl::hal::device_output &output, const_ptr addr, size_t count);

    error_dsm flush_dirty(__impl::dsm::address_space_ptr aspace);

    //////////////////////
    // Helper functions //
    //////////////////////
    bool
    helper_insert(__impl::hal::aspace &as, mapping_ptr m)
    {
        error_dsm ret;

        parent::map_mapping_group &group = parent::get_aspace_mappings(as);
        ret = parent::insert_mapping(group, m);

        return ret == error_dsm::DSM_SUCCESS;
    }

    bool
    helper_delete_mappings(__impl::hal::aspace &as)
    {
        parent::map_mapping_group &group = parent::get_aspace_mappings(as);

        return parent::delete_mappings(group) == error_dsm::DSM_SUCCESS;
    }

    parent::map_mapping &
    helper_get_mappings(__impl::hal::aspace &as, ptr::backend_type p)
    {
        parent::map_mapping_group &group = parent::get_aspace_mappings(as);

        parent::map_mapping_group::iterator it = group.find(p);
        ASSERTION(it != group.end());

        return it->second;
    }

    __impl::dsm::mapping_ptr
    helper_get_mapping(ptr p)
    {
        parent::map_mapping_group &group = parent::get_aspace_mappings(*p.get_aspace());

        parent::range_mapping range = parent::get_mappings_in_range<false>(group, p, 1);
        if (range.is_empty()) {
            return NULL;
        }

        return *range.begin();
    }
};
