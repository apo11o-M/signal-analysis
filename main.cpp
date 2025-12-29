#include <iostream>

// #include "Transmitter.hpp"
// #include "Receiver.hpp"
#include "TransmitterSingleTone.hpp"
#include "ReceiverSingleTone.hpp"
#include "TransmitterChirp.hpp"
#include "ReceiverChirp.hpp"

#include "Channel.hpp"
#include "Logger.hpp"
#include "DataWriter.hpp"

#define FRAME_COUNT 100

int main() {
    std::cout << "Signal analysis executable starting.." << std::endl;

    // 1. parse config file
    // 2. create transmitter, impairments, and receiver objects
    // 3. go through the sim
    // 4. visualization

    Logger logger("sim.log");

    // TxConfigSingleTone tx_config;
    // std::unique_ptr<Transmitter> tx_ptr = std::make_unique<TransmitterSingleTone>(tx_config);
    TxConfigChirp tx_config;
    std::unique_ptr<Transmitter> tx_ptr = std::make_unique<TransmitterChirp>(tx_config);
    logger.log(Logger::Level::INFO, "Transmitter initialized.");

    // RxConfigSingleTone rx_config;
    // std::unique_ptr<Receiver> rx_ptr = std::make_unique<ReceiverSingleTone>(rx_config);
    RxConfigChirp rx_config;
    std::unique_ptr<Receiver> rx_ptr = std::make_unique<ReceiverChirp>(rx_config);
    logger.log(Logger::Level::INFO, "Receiver initialized.");

    Channel channel(rx_config.common.sample_rate_hz, logger);
    channel.add<GainImpairment>(1.0f);
    channel.add<CFOImpairment>(250.0, 0.0);
    channel.add<PhaseOffsetImpairment>(0.3);
    channel.add<TimingOffsetImpairment>(7, TimingOffsetImpairment::Mode::Circular);
    channel.add<AWGNImpairment>(20.0);
    logger.log(Logger::Level::INFO, "Channel initialized.");

    DataWriter writer("dumps", "basic_tx");
    logger.log(Logger::Level::INFO, "DataWriter initialized at " + writer.run_dir().string());

    for (uint64_t iter = 0; iter < FRAME_COUNT; iter++) {
        Frame f = tx_ptr->next_frame();

        f.dump(logger.stream(Logger::Level::INFO), 5);
        writer.write_iq_frame_binary("tx", f);

        // channel.apply(f);
        // f.dump(logger.stream(Logger::Level::INFO), 5);
        // writer.write_iq_frame_binary("imp", f);

        std::unique_ptr<RxResults> rx_res = rx_ptr->process_frame(f);
        writer.write_rx_results(rx_res);

    }

    std::cout << "Signal analysis executable finished" << std::endl;
}
