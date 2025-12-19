#include <iostream>

#include "Transmitter.hpp"
#include "Receiver.hpp"
#include "Logger.hpp"
#include "DataWriter.hpp"

int main() {
    std::cout << "Signal analysis executable starting.." << std::endl;

    // 1. parse config file
    // 2. create transmitter, impairments, and receiver objects
    // 3. go through the sim
    // 4. visualization

    TxConfig tx_config;
    Transmitter tx(tx_config);

    Logger logger("sim.log");
    logger.log(Logger::Level::INFO, "Transmitter initialized.");

    DataWriter writer("dumps", "basic_tx");
    logger.log(Logger::Level::INFO, "DataWriter initialized at " + writer.run_dir().string());

    for (uint64_t iter = 0; iter < 20; iter++) {
        Frame f = tx.next_frame();

        f.dump(logger.stream(Logger::Level::INFO), 5);
        writer.write_frame("tx", f);

    }

    std::cout << "Signal analysis executable finished" << std::endl;
}
