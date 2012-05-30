#include "unit2/hal/init.h"

#include "hal/types.h"

#include "util/misc.h"

#include "gtest/gtest.h"

namespace I_HAL = __impl::hal;

using I_HAL::ptr;

static I_HAL::phys::list_platform platforms;

#define ASSERT_HAL_SUCCESS(e) do { if (e != I_HAL::error::HAL_SUCCESS) { printf("HAL error: %d\n", e); }  ASSERT_TRUE(e == I_HAL::error::HAL_SUCCESS); } while (0)

void
hal_init_test::SetUpTestCase()
{
    // Inititalize platform
    I_HAL::error err = I_HAL::init();
    ASSERT_HAL_SUCCESS(err);

    // Get platforms
    platforms = I_HAL::phys::get_platforms();
    ASSERT_TRUE(platforms.size() > 0);
}

void
hal_init_test::TearDownTestCase()
{
    I_HAL::error err;

    err = I_HAL::fini();
    ASSERT_HAL_SUCCESS(err);
}

TEST_F(hal_init_test, platform)
{
    ASSERT_TRUE(platforms.size() > 0);

    for (auto plat : platforms) {
        I_HAL::phys::platform::set_memory memories;
        I_HAL::phys::platform::set_aspace paspaces;

        I_HAL::phys::platform::set_processing_unit pUnits;

        memories = plat->get_memories();
        ASSERT_TRUE(memories.size() > 0);
        paspaces = plat->get_paspaces();
        ASSERT_TRUE(paspaces.size() > 0);

        pUnits = plat->get_processing_units();
        ASSERT_TRUE(pUnits.size() > 0);
    }
}

TEST_F(hal_init_test, aspace)
{
    ASSERT_TRUE(platforms.size() > 0);

    for (auto plat : platforms) {
        for (auto pUnit : plat->get_processing_units()) {
            I_HAL::phys::aspace &pas = pUnit->get_paspace();

            I_HAL::virt::aspace *as0;
            I_HAL::virt::aspace *as1;

            I_HAL::error err;

            I_HAL::phys::aspace::set_processing_unit compatibleUnits({ pUnit });

            // Create address spaces on the physical address space
            as0 = pas.create_vaspace(compatibleUnits, err);
            ASSERT_HAL_SUCCESS(err);
            as1 = pas.create_vaspace(compatibleUnits, err);
            ASSERT_HAL_SUCCESS(err);

            // Create address spaces on the physical address space
            err = pas.destroy_vaspace(*as0);
            ASSERT_HAL_SUCCESS(err);
            err = pas.destroy_vaspace(*as1);
            ASSERT_HAL_SUCCESS(err);
        }
    }
}


template <enum I_HAL::phys::processing_unit::type PUnit>
void do_object_alloc()
{
    static const unsigned ELEMS = 1024;

    ASSERT_TRUE(platforms.size() > 0);

    I_HAL::phys::platform *plat0;
    plat0 = platforms.front();
    // Get processing units
    I_HAL::phys::platform::set_processing_unit pUnits;
    pUnits = plat0->get_processing_units(PUnit);

    I_HAL::phys::processing_unit *pUnit0;
    I_HAL::phys::aspace *pas0;

    I_HAL::virt::aspace *as0;

    I_HAL::virt::object *obj0;

    I_HAL::error err;

    ASSERT_TRUE(pUnits.size() > 0);
    pUnit0 = *pUnits.begin();
    pas0 = &pUnit0->get_paspace();
    // Create address spaces
    I_HAL::phys::aspace::set_processing_unit compatibleUnits({ pUnit0 });
    as0 = pas0->create_vaspace(compatibleUnits, err);
    ASSERT_HAL_SUCCESS(err);

    obj0 = plat0->create_object(**pas0->get_memories().begin(), ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);

    ptr ptrBase = as0->map(*obj0, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);

    if (PUnit == I_HAL::phys::processing_unit::type::PUNIT_TYPE_CPU) {
        float (*mat)[ELEMS] = (float (*)[ELEMS]) ptrBase.get_view().get_offset() + ptrBase.get_offset();
        (*mat)[0]         = 0.f;
        (*mat)[ELEMS - 1] = 0.f;
    }

    err = as0->unmap(ptrBase);
    ASSERT_HAL_SUCCESS(err);

    err = plat0->destroy_object(*obj0);
    ASSERT_HAL_SUCCESS(err);

    err = pas0->destroy_vaspace(*as0);
    ASSERT_HAL_SUCCESS(err);
}

void do_object_map_host()
{
    static const unsigned ELEMS = 1024;

    ASSERT_TRUE(platforms.size() > 0);

    I_HAL::phys::platform *plat0;
    plat0 = platforms.front();
    // Get processing units
    I_HAL::phys::platform::set_processing_unit pUnits;

    ASSERT_TRUE(plat0->get_processing_units(I_HAL::phys::processing_unit::type::PUNIT_TYPE_CPU).size () > 0);

    I_HAL::phys::processing_unit *pUnit0;
    I_HAL::phys::aspace *pas0;

    I_HAL::virt::aspace *as0;

    I_HAL::virt::object *obj0;

    I_HAL::error err;

    I_HAL::phys::platform::set_memory mems = plat0->get_memories();

    pUnit0 = *plat0->get_processing_units(I_HAL::phys::processing_unit::type::PUNIT_TYPE_CPU).begin();

    pas0 = &pUnit0->get_paspace();

    // Create address spaces
    I_HAL::phys::aspace::set_processing_unit compatibleUnits({ pUnit0 });
    as0 = pas0->create_vaspace(compatibleUnits, err);
    ASSERT_HAL_SUCCESS(err);

    obj0 = plat0->create_object(pUnit0->get_preferred_memory(), ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);

    ptr ptrBase0 = as0->map(*obj0, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);
    ptr ptrBase1 = as0->map(*obj0, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);

    float (&mat1)[ELEMS] = *((float (*)[ELEMS]) ptrBase0.get_view().get_offset() + ptrBase0.get_offset());
    for (unsigned i = 0; i < ELEMS; ++i) {
        mat1[i] = float(i);
    }

    float (&mat2)[ELEMS] = *((float (*)[ELEMS]) ptrBase1.get_view().get_offset() + ptrBase1.get_offset());
    for (unsigned i = 0; i < ELEMS; ++i) {
        ASSERT_TRUE(mat1[i] == mat2[i]);
    }

    err = as0->unmap(ptrBase0);
    ASSERT_HAL_SUCCESS(err);
    err = as0->unmap(ptrBase1);
    ASSERT_HAL_SUCCESS(err);

    err = plat0->destroy_object(*obj0);
    ASSERT_HAL_SUCCESS(err);

    err = pas0->destroy_vaspace(*as0);
    ASSERT_HAL_SUCCESS(err);
}

void do_object_copy_host()
{
    static const unsigned ELEMS = 1024;

    ASSERT_TRUE(platforms.size() > 0);

    I_HAL::phys::platform *plat0;
    plat0 = platforms.front();
    // Get processing units
    I_HAL::phys::platform::set_processing_unit pUnits;

    pUnits = plat0->get_processing_units(I_HAL::phys::processing_unit::type::PUNIT_TYPE_CPU);
    ASSERT_TRUE(pUnits.size () > 0);

    I_HAL::phys::processing_unit *pUnit0;
    I_HAL::phys::aspace *pas0;

    I_HAL::virt::aspace *as0, *as1;

    I_HAL::virt::object *obj0, *obj1;

    I_HAL::error err;

    pUnit0 = *pUnits.begin();

    pas0 = &pUnit0->get_paspace();

    // Create address spaces
    I_HAL::phys::aspace::set_processing_unit compatibleUnits({ pUnit0 });
    as0 = pas0->create_vaspace(compatibleUnits, err);
    ASSERT_HAL_SUCCESS(err);
    as1 = as0;

    obj0 = plat0->create_object(**pas0->get_memories().begin(), ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);
    obj1 = plat0->create_object(**pas0->get_memories().begin(), ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);

    ptr ptrBase0 = as0->map(*obj0, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);
    ptr ptrBase1 = as1->map(*obj1, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);

    float (&mat1)[ELEMS] = *((float (*)[ELEMS]) ptrBase0.get_view().get_offset() + ptrBase0.get_offset());
    for (unsigned i = 0; i < ELEMS; ++i) {
        mat1[i] = float(i);
    }

    I_HAL::copy(ptrBase1, ptrBase0, ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);

    float (&mat2)[ELEMS] = *((float (*)[ELEMS]) ptrBase1.get_view().get_offset() + ptrBase1.get_offset());
    for (unsigned i = 0; i < ELEMS; ++i) {
        ASSERT_TRUE(mat1[i] == mat2[i]);
    }

    err = as0->unmap(ptrBase0);
    ASSERT_HAL_SUCCESS(err);
    err = as1->unmap(ptrBase1);
    ASSERT_HAL_SUCCESS(err);

    err = plat0->destroy_object(*obj0);
    ASSERT_HAL_SUCCESS(err);
    err = plat0->destroy_object(*obj1);
    ASSERT_HAL_SUCCESS(err);

    err = pas0->destroy_vaspace(*as0);
    ASSERT_HAL_SUCCESS(err);
    if (as1 != as0) {
        err = pas0->destroy_vaspace(*as1);
        ASSERT_HAL_SUCCESS(err);
    }
}

void do_object_copy_gpu()
{
    static const unsigned ELEMS = 1024;

    ASSERT_TRUE(platforms.size() > 0);

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

    I_HAL::error err;

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

    objCPU0 = plat0->create_object(pUnitCPU->get_preferred_memory(), ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);
    objCPU1 = plat0->create_object(pUnitCPU->get_preferred_memory(), ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);
    objGPU  = plat0->create_object(pUnitGPU->get_preferred_memory(), ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);

    ptr ptrBaseCPU0 = asCPU->map(*objCPU0, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);
    ptr ptrBaseCPU1 = asCPU->map(*objCPU1, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);
    ptr ptrBaseGPU  = asGPU->map(*objGPU, GMAC_PROT_READWRITE, err);
    ASSERT_HAL_SUCCESS(err);

    float (&mat1)[ELEMS] = *((float (*)[ELEMS]) ptrBaseCPU0.get_view().get_offset() + ptrBaseCPU0.get_offset());
    for (unsigned i = 0; i < ELEMS; ++i) {
        mat1[i] = float(i);
    }

    I_HAL::copy(ptrBaseGPU, ptrBaseCPU0, ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);

    I_HAL::copy(ptrBaseCPU1, ptrBaseGPU, ELEMS * sizeof(float), err);
    ASSERT_HAL_SUCCESS(err);

    float (&mat2)[ELEMS] = *((float (*)[ELEMS]) ptrBaseCPU1.get_view().get_offset() + ptrBaseCPU1.get_offset());
    for (unsigned i = 0; i < ELEMS; ++i) {
        ASSERT_TRUE(mat1[i] == mat2[i]);
    }

    err = asCPU->unmap(ptrBaseCPU0);
    ASSERT_HAL_SUCCESS(err);
    err = asCPU->unmap(ptrBaseCPU1);
    ASSERT_HAL_SUCCESS(err);
    err = asGPU->unmap(ptrBaseGPU);
    ASSERT_HAL_SUCCESS(err);

    err = plat0->destroy_object(*objCPU0);
    ASSERT_HAL_SUCCESS(err);
    err = plat0->destroy_object(*objCPU1);
    ASSERT_HAL_SUCCESS(err);
    err = plat0->destroy_object(*objGPU);
    ASSERT_HAL_SUCCESS(err);

    err = pasCPU->destroy_vaspace(*asCPU);
    ASSERT_HAL_SUCCESS(err);
    err = pasGPU->destroy_vaspace(*asGPU);
    ASSERT_HAL_SUCCESS(err);
}

TEST_F(hal_init_test, object_alloc)
{
    do_object_alloc<
                    I_HAL::phys::processing_unit::type::PUNIT_TYPE_CPU
                   >();

    do_object_alloc<
                    I_HAL::phys::processing_unit::type::PUNIT_TYPE_GPU
                   >();
}

TEST_F(hal_init_test, object_map_host)
{
    do_object_map_host();
}


TEST_F(hal_init_test, object_copy_host)
{
    do_object_copy_host();
}

TEST_F(hal_init_test, object_copy_gpu)
{
    do_object_copy_gpu();
}



#if 0
    static I_HAL::phys::processing_unit *pUnit0;
    static I_HAL::phys::processing_unit *pUnit1;
    static I_HAL::phys::aspace *pas0;
    static I_HAL::virt::aspace *as0;
    static I_HAL::virt::aspace *as1;

    // Get processing units
    pUnits = plat0->get_processing_units(I_HAL::phys::processing_unit::PUNIT_TYPE_GPU);
    ASSERT_TRUE(pUnits.size() > 0);
    pUnit0 = *pUnits.begin();
    pUnit1 = nullptr;
    pas0 = &pUnit0->get_paspace();
    // Create address spaces
    I_HAL::phys::aspace::set_processing_unit compatibleUnits({ pUnit0 });
    as0 = pas0->create_vaspace(compatibleUnits, err);
    ASSERT_HAL_SUCCESS(err);
    as1 = pas0->create_vaspace(compatibleUnits, err);
    ASSERT_HAL_SUCCESS(err);

    err = pas0->destroy_vaspace(*as0);
    ASSERT_HAL_SUCCESS(err);
    err = pas0->destroy_vaspace(*as1);
    ASSERT_HAL_SUCCESS(err);
#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
