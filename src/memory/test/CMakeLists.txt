set(memory_test_SRC
    SharedObject.h SharedObject.ipp
    ReplicatedObject.h ReplicatedObject.ipp)

add_gmac_sources(gmac-dbc ${memory_test_SRC})
