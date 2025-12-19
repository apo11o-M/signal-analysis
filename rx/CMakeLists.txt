
set(CXX_SOURCES )

add_library(signal-rx STATIC ${CXX_SOURCES})

target_include_directories(signal-rx PUBLIC include)
