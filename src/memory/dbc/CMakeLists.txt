set(memory_test_SRC
    Manager.h Manager.cpp
    SharedObject.h SharedObject.ipp
    ReplicatedObject.h ReplicatedObject.ipp
    Block.h Block.ipp
    AcceleratorBlock.h AcceleratorBlock.ipp
    SystemBlock.h SystemBlock.ipp
)

add_gmac_sources(gmac-dbc ${memory_test_SRC})
