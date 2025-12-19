
set(CXX_SOURCES )

add_library(signal-tx STATIC ${CXX_SOURCES})

target_include_directories(signal-tx PUBLIC include)