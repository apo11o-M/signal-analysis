
set(CXX_SOURCES )

add_library(signal-utils STATIC ${CXX_SOURCES})

target_include_directories(signal-utils PUBLIC include)
