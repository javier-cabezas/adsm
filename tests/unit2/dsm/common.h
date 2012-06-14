#ifndef GMAC_TEST_UNIT_COMMON_H_
#define GMAC_TEST_UNIT_COMMON_H_

#define I_DSM  __impl::dsm
#define I_HAL  __impl::hal
#define I_UTIL __impl::util

#define ASSERT_DSM_SUCCESS(e) do { if (e != I_DSM::error::DSM_SUCCESS) { printf("DSM error: %d\n", e); }  ASSERT_TRUE(e == I_DSM::error::DSM_SUCCESS); } while (0)
#define ASSERT_DSM_FAILURE(e) do { if (e == I_DSM::error::DSM_SUCCESS) { printf("DSM expected error, got SUCCESS\n"); } ASSERT_TRUE(e != I_DSM::error::DSM_SUCCESS); } while (0)

#endif
