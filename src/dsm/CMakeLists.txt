set(dsm_SRC
    coherence.h
    manager.h
    manager-impl.h
    manager.cpp
    mapping.h
    mapping-impl.h
    mapping.cpp
    coherence/block.h
    coherence/block-impl.h
    coherence/types.h
)

add_gmac_sources(gmac-dsm ${dsm_SRC})
