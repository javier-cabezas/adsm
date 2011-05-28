set(arch_hpe_gpu_amd_SRC
    Accelerator.h
    Accelerator.cpp
)

set(arch_hpe_gpu_amd_DBC
)

add_gmac_sources(gmac-arch-hpe-gpu-amd ${arch_hpe_gpu_amd_SRC} ${arch_hpe_gpu_amd_DBC})
