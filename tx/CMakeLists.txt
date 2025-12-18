
set(CXX_SOURCES src/tx_main.cpp)

add_library(signal-tx STATIC ${CXX_SOURCES})

target_include_directories(signal-tx PUBLIC include)