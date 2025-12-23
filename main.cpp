#include <iostream>

#include "Transmitter.hpp"
#include "Receiver.hpp"
#include "Channel.hpp"
#include "Logger.hpp"
#include "DataWriter.hpp"

int main() {
    std::cout << "Signal analysis executable starting.." << std::endl;

    // 1. parse config file
    // 2. create transmitter, impairments, and receiver objects
    // 3. go through the sim
    // 4. visualization

    Logger logger("sim.log");

    TxConfig tx_config;
    Transmitter tx(tx_config);
    logger.log(Logger::Level::INFO, "Transmitter initialized.");

    RxConfig rx_config;
    Receiver rx(rx_config);
    logger.log(Logger::Level::INFO, "Receiver initialized.");

    Channel channel(rx_config.sample_rate_hz, logger);
    channel.add<GainImpairment>(1.0f);
    channel.add<CFOImpairment>(250.0, 0.0);
    channel.add<AWGNImpairment>(20.0);
    logger.log(Logger::Level::INFO, "Channel initialized.");

    DataWriter writer("dumps", "basic_tx");
    logger.log(Logger::Level::INFO, "DataWriter initialized at " + writer.run_dir().string());

    for (uint64_t iter = 0; iter < 20; iter++) {
        Frame f = tx.next_frame();

        f.dump(logger.stream(Logger::Level::INFO), 5);
        writer.write_iq_frame_binary("tx", f);

        channel.apply(f);
        f.dump(logger.stream(Logger::Level::INFO), 5);
        writer.write_iq_frame_binary("imp", f);

        RxResults rx_res = rx.process_frame(f);
        writer.write_rx_results(rx_res);

    }

    std::cout << "Signal analysis executable finished" << std::endl;
}
