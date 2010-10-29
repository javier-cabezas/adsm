set(memory_test_SRC
    Manager.h
    SharedObject.h SharedObject.ipp
    ReplicatedObject.h ReplicatedObject.ipp)

add_gmac_sources(gmac-dbc ${memory_test_SRC})
