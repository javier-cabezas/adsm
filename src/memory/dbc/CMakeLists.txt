set(memory_test_SRC
    Manager.h
    SharedObject.h SharedObject.ipp
    ReplicatedObject.h ReplicatedObject.ipp
    Block.h AcceleratorBlock.h SystemBlock.h
)

add_gmac_sources(gmac-dbc ${memory_test_SRC})
