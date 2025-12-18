#include <iostream>

#include "Transmitter.hpp"

int main() {
    std::cout << "Program Starts.." << std::endl;

    // 1. parse config file
    // 2. create transmitter, impairments, and receiver objects
    // 3. go through the sim
    // 4. visualization

    TxConfig tx_config;
    Transmitter tx(tx_config);

    for (uint64_t iter = 0; iter < 3; iter++) {
        Frame f = tx.next_frame();
        f.dump();
    }

    std::cout << "..Finished" << std::endl;
}
