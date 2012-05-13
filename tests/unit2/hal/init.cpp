#include "unit2/hal/init.h"

#include "hal/types.h"

#include "util/misc.h"

#include "gtest/gtest.h"

namespace I_HAL = __impl::hal;

static I_HAL::phys::list_platform platforms;
static I_HAL::phys::platform *plat0;
static I_HAL::phys::platform::set_memory memories;
static I_HAL::phys::platform::set_processing_unit pUnits;
static I_HAL::phys::processing_unit *pUnit0;
static I_HAL::phys::processing_unit *pUnit1;
static I_HAL::phys::aspace *pas0;
static I_HAL::virt::aspace *as0;
static I_HAL::virt::aspace *as1;

#if 0
static I_HAL::virt::object *obj0;
static I_HAL::virt::object *obj1;
#endif

void
hal_init_test::SetUpTestCase()
{
    // Inititalize platform
    I_HAL::error err = I_HAL::init();
    ASSERT_TRUE(err == I_HAL::error::HAL_SUCCESS);
    // Get platforms
    platforms = I_HAL::phys::get_platforms();
    ASSERT_TRUE(platforms.size() > 0);
    plat0 = platforms.front();
    // Get processing units
    pUnits = plat0->get_processing_units(I_HAL::phys::processing_unit::PUNIT_TYPE_GPU);
    ASSERT_TRUE(pUnits.size() > 0);
    pUnit0 = *pUnits.begin();
    pUnit1 = nullptr;
    pas0 = &pUnit0->get_paspace();
    // Create address spaces
    I_HAL::phys::aspace::set_processing_unit compatibleUnits({ pUnit0 });
    as0 = pas0->create_vaspace(compatibleUnits, err);
    ASSERT_TRUE(err == I_HAL::error::HAL_SUCCESS);
    as1 = pas0->create_vaspace(compatibleUnits, err);
    ASSERT_TRUE(err == I_HAL::error::HAL_SUCCESS);
}

void
hal_init_test::TearDownTestCase()
{
    I_HAL::error err;

    err = pas0->destroy_vaspace(*as0);
    ASSERT_TRUE(err == I_HAL::error::HAL_SUCCESS);
    err = pas0->destroy_vaspace(*as1);
    ASSERT_TRUE(err == I_HAL::error::HAL_SUCCESS);

    I_HAL::fini();
}

TEST_F(hal_init_test, init)
{
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
