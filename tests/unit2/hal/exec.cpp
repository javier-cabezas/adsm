#include "unit2/hal/exec.h"

#include "hal/types.h"

#include "util/file.h"
#include "util/misc.h"

#include "gtest/gtest.h"

namespace I_HAL = __impl::hal;

using I_HAL::ptr;

static I_HAL::phys::list_platform platforms;

#define ASSERT_HAL_SUCCESS(e) do { if (e != I_HAL::error::HAL_SUCCESS) { printf("HAL error: %d\n", e); }  ASSERT_TRUE(e == I_HAL::error::HAL_SUCCESS); } while (0)

void
hal_exec_test::SetUpTestCase()
{
    // Inititalize platform
    I_HAL::error err = I_HAL::init();
    ASSERT_HAL_SUCCESS(err);

    // Get platforms
    platforms = I_HAL::phys::get_platforms();
    ASSERT_TRUE(platforms.size() > 0);
}

void
hal_exec_test::TearDownTestCase()
{
    I_HAL::error err;

    err = I_HAL::fini();
    ASSERT_HAL_SUCCESS(err);
}

enum source_type {
    SOURCE_FILE,
    SOURCE_STRING,
    SOURCE_HANDLE
};

template <bool Sync, source_type Type>
void do_exec_gpu(const void *ptrSource)
{
    // Constants
    static const unsigned ELEMS = 1024;
    static const unsigned ITER  = 1024;

    ASSERT_TRUE(platforms.size() > 0);

    I_HAL::error err;

    I_HAL::code::repository repo;
    I_HAL::code::repository_view *repoView;
    const I_HAL::code::kernel_t *kernel;

    // Load device code
    switch (Type) {
    case source_type::SOURCE_FILE:
        err = repo.load_from_file((const char *) ptrSource, "");
        break;
    case source_type::SOURCE_STRING:
        err = repo.load_from_mem((const char *) ptrSource, strlen((const char *) ptrSource), "");
        break;
    case source_type::SOURCE_HANDLE:
        err = repo.load_from_handle(ptrSource, "");
        break;
    }
    ASSERT_HAL_SUCCESS(err);

    I_HAL::phys::platform *plat0;
    plat0 = platforms.front();
    // Get processing units
    I_HAL::phys::platform::set_processing_unit pUnitsCPU, pUnitsGPU;

    pUnitsCPU = plat0->get_processing_units(I_HAL::phys::processing_unit::type::PUNIT_TYPE_CPU);
    ASSERT_TRUE(pUnitsCPU.size () > 0);
    pUnitsGPU = plat0->get_processing_units(I_HAL::phys::processing_unit::type::PUNIT_TYPE_GPU);
    ASSERT_TRUE(pUnitsGPU.size () > 0);

    I_HAL::phys::processing_unit *pUnitCPU, *pUnitGPU;
    I_HAL::phys::aspace *pasCPU, *pasGPU;

    I_HAL::virt::aspace *asCPU, *asGPU;

    I_HAL::virt::object *objCPU0, *objCPU1, *objGPU;

    pUnitCPU = *pUnitsCPU.begin();
    pUnitGPU = *pUnitsGPU.begin();

    pasCPU = &pUnitCPU->get_paspace();
    pasGPU = &pUnitGPU->get_paspace();

    // Create address spaces
    I_HAL::phys::aspace::set_processing_unit compatibleUnitsCPU({ pUnitCPU });
    I_HAL::phys::aspace::set_processing_unit compatibleUnitsGPU({ pUnitGPU });
    asCPU = pasCPU->create_vaspace(compatibleUnitsCPU, err);
    ASSERT_HAL_SUCCESS(err);
    asGPU = pasGPU->create_vaspace(compatibleUnitsGPU, err);
    ASSERT_HAL_SUCCESS(err);

    I_HAL::virt::context *ctx = asGPU->create_context(err);
    ASSERT_HAL_SUCCESS(err);

    // Map code and get kernel handler
    repoView = asGPU->map(repo, err);
    ASSERT_HAL_SUCCESS(err);

    kernel = repoView->get_kernel("inc");
    ASSERT_TRUE(kernel != nullptr);

    // Create data objects
    objCPU0 = plat0->create_object(pUnitCPU->get_preferred_memory(), ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);
    objCPU1 = plat0->create_object(pUnitCPU->get_preferred_memory(), ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);
    objGPU  = plat0->create_object(pUnitGPU->get_preferred_memory(), ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);

    // Map objects on address spaces
    ptr ptrBaseCPU0 = asCPU->map(*objCPU0, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);
    ptr ptrBaseCPU1 = asCPU->map(*objCPU1, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);
    ptr ptrBaseGPU  = asGPU->map(*objGPU, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);

    // Initialize data
    float (&mat1)[ELEMS] = *((float (*)[ELEMS]) ptrBaseCPU0.get_view().get_offset() + ptrBaseCPU0.get_offset());
    for (unsigned i = 0; i < ELEMS; ++i) {
        mat1[i] = float(i);
    }

    // Copy to GPU
    I_HAL::copy(ptrBaseGPU, ptrBaseCPU0, ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);

    // Configure kernel
    void *arg = (void *) (ptrBaseGPU.get_view().get_offset() + ptrBaseGPU.get_offset());
    I_HAL::code::kernel_config conf(ELEMS/256, 256, 0, 0);
    I_HAL::code::kernel_args args;
    err = args.push_arg(&arg, sizeof(void *));
    ASSERT_HAL_SUCCESS(err);

    // Execute kernel ITER times
    I_HAL::event_ptr evt;
    for (unsigned it = 0; it < ITER; ++it) {
        evt = ctx->queue(*kernel, conf, args, err);
        ASSERT_HAL_SUCCESS(err);
        if (Sync) {
            err = evt->sync();
            ASSERT_HAL_SUCCESS(err);
        }
    }

    // Copy data back to host
    if (Sync) {
        evt = I_HAL::copy(ptrBaseCPU1, ptrBaseGPU, ELEMS * sizeof(float), err);
    } else {
        evt = I_HAL::copy(ptrBaseCPU1, ptrBaseGPU, ELEMS * sizeof(float), evt, err);
    }
    ASSERT_HAL_SUCCESS(err);

    // Check results
    float (&mat2)[ELEMS] = *((float (*)[ELEMS]) ptrBaseCPU1.get_view().get_offset() + ptrBaseCPU1.get_offset());
    for (unsigned i = 0; i < ELEMS; ++i) {
        ASSERT_TRUE((mat1[i] + float(ITER)) == mat2[i]);
    }

    // Unmap objects
    err = asCPU->unmap(ptrBaseCPU0);
    ASSERT_HAL_SUCCESS(err);
    err = asCPU->unmap(ptrBaseCPU1);
    ASSERT_HAL_SUCCESS(err);
    err = asGPU->unmap(ptrBaseGPU);
    ASSERT_HAL_SUCCESS(err);

    // Unmap code
    err = asGPU->unmap(*repoView);
    ASSERT_HAL_SUCCESS(err);

    // Destroy data objects
    err = plat0->destroy_object(*objCPU0);
    ASSERT_HAL_SUCCESS(err);
    err = plat0->destroy_object(*objCPU1);
    ASSERT_HAL_SUCCESS(err);
    err = plat0->destroy_object(*objGPU);
    ASSERT_HAL_SUCCESS(err);

    // Destroy address spaces
    err = pasCPU->destroy_vaspace(*asCPU);
    ASSERT_HAL_SUCCESS(err);
    err = pasGPU->destroy_vaspace(*asGPU);
    ASSERT_HAL_SUCCESS(err);
}

TEST_F(hal_exec_test, exec_gpu_file)
{
    do_exec_gpu<true,  source_type::SOURCE_FILE>("code/common.lib");
    do_exec_gpu<false, source_type::SOURCE_FILE>("code/common.lib");
}

TEST_F(hal_exec_test, exec_gpu_string)
{
    gmacError_t err;

    std::string file;
    file = __impl::util::get_file_contents("code/common.lib", err);
    ASSERT_TRUE(err == gmacSuccess);
    
    do_exec_gpu<true,  source_type::SOURCE_STRING>(file.c_str());
    do_exec_gpu<false, source_type::SOURCE_STRING>(file.c_str());
}

#if 0
TEST_F(hal_exec_test, exec_gpu_handle)
{
    do_exec_gpu<true,  source_type::SOURCE_HANDLE>("code/common.lib");
    do_exec_gpu<false, source_type::SOURCE_HANDLE>("code/common.lib");
}
#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
