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
#include "SimConfig.hpp"

int main(int argc, char* argv[]) {
    std::cout << "Signal analysis executable starting.." << std::endl;

    Logger logger("sim.log");

    std::string config_filename = "../sim_config/chirp.json";
    if (argc >= 2) { config_filename = std::string(argv[1]); }
    SimConfig simConfig(config_filename);
    simConfig.validateJson(logger);
    logger.log(Logger::Level::INFO, "Simulation config imported");

    std::unique_ptr<Transmitter> tx_ptr;
    std::unique_ptr<Receiver> rx_ptr;
    const uint64_t frameCount = simConfig.at("frame_count").get<uint64_t>();
    const uint64_t frameSize = simConfig.at("frame_size").get<uint64_t>();

    TxConfigCommon tx_common_config;
    tx_common_config.frame_len = frameSize;
    tx_common_config.sample_rate_hz = simConfig.at("tx").at("sample_rate_hz").get<double>();

    RxConfigCommon rx_common_config;
    rx_common_config.sample_rate_hz = simConfig.at("rx").at("sample_rate_hz").get<double>();
    rx_common_config.snr_threshold_db = simConfig.at("rx").at("snr_threshold_db").get<float>();


    // Transmitter initialization
    if (simConfig.at("tx").at("tx_type") == "SingleTone") {
        std::cout << "Transmitter type: SingleTone" << std::endl;
        logger.log(Logger::Level::INFO, "Transmitter type: SingleTone");

        TxConfigSingleTone tx_config;
        tx_config.common = tx_common_config;
        // TODO: load single tone transmitter json configs
        tx_ptr = std::make_unique<TransmitterSingleTone>(tx_config);
    }
    else if (simConfig.at("tx").at("tx_type") == "Chirp") {
        std::cout << "Transmitter type: Chirp" << std::endl;
        logger.log(Logger::Level::INFO, "Transmitter type: Chirp");

        TxConfigChirp tx_config;
        tx_config.common = tx_common_config;
        tx_config.f0 = simConfig.at("tx").at("f0").get<double>();
        tx_config.chirp_rate_hz = simConfig.at("tx").at("chirp_rate_hz").get<double>();
        tx_config.duration_sample = simConfig.at("tx").at("duration_sample").get<std::size_t>();
        tx_config.sweep_direction = simConfig.at("tx").at("sweep_direction").get<int8_t>();
        tx_config.amplitude = simConfig.at("tx").at("amplitude").get<float>();
        tx_ptr = std::make_unique<TransmitterChirp>(tx_config);
    }
    else {
        std::cout << "Transmitter type: Unknown.. \nAbort" << std::endl;
        logger.log(Logger::Level::ERROR, "Transmitter type: Unknown.. \nAbort");
        return -1;
    }
    logger.log(Logger::Level::INFO, "Transmitter initialized.");


    // Receiver initialization
    if (simConfig.at("rx").at("rx_type") == "SingleTone") {
        std::cout << "Receiver type: SingleTone" << std::endl;
        logger.log(Logger::Level::INFO, "Receiver type: SingleTone");

        RxConfigSingleTone rx_config;
        rx_config.common = rx_common_config;
        // TODO: load single tone receiver json configs
        rx_ptr = std::make_unique<ReceiverSingleTone>(rx_config);
    }
    else if (simConfig.at("rx").at("rx_type") == "Chirp") {
        std::cout << "Receiver type: Chirp" << std::endl;
        logger.log(Logger::Level::INFO, "Receiver type: Chirp");

        RxConfigChirp rx_config;
        rx_config.common = rx_common_config;
        rx_config.frame_len = frameSize;
        // rx_config.chirp_start_hz = simConfig.at("rx").at("chirp_start_hz").get<double>();
        // rx_config.chirp_end_hz = simConfig.at("rx").at("chirp_end_hz").get<double>();
        // rx_config.search_max_offset_samples = simConfig.at("rx").at("search_max_offset_samples").get<std::size_t>();
        // rx_config.detection_threshold = simConfig.at("rx").at("detection_threshold").get<double>();
        // rx_config.cfo_est_skip_samples = simConfig.at("rx").at("cfo_est_skip_samples").get<std::size_t>();
        // rx_config.cfo_est_count_samples = simConfig.at("rx").at("cfo_est_count_samples").get<std::size_t>();
        rx_ptr = std::make_unique<ReceiverChirp>(rx_config);
    }
    else {
        std::cout << "Receiver type: Unknown.. \nAbort" << std::endl;
        logger.log(Logger::Level::ERROR, "Receiver type: Unknown.. \nAbort");
        return -1;
    }
    logger.log(Logger::Level::INFO, "Receiver initialized.");


    // Channel initialization
    Channel channel(tx_ptr->getSampleRateHz(), logger);
    std::vector<std::string> imp_types = simConfig.at("imp").at("imp_types").get<std::vector<std::string>>();
    for (auto& s : imp_types) {
        logger.log(Logger::Level::INFO, "    Impairment: " + s);
        if (s == "gain") {
            const float gain = simConfig.at("imp").at("gain").at("gain").get<float>();
            channel.add<GainImpairment>(gain);
        }
        else if (s == "cfo") {
            const double cfo_hz = simConfig.at("imp").at("cfo").at("cfo_hz").get<double>();
            const double initial_phase_rad = simConfig.at("imp").at("cfo").at("initial_phase_rad").get<double>();
            channel.add<CFOImpairment>(cfo_hz, initial_phase_rad);
        }
        else if (s == "phase_offset") {
            const double phase_rad = simConfig.at("imp").at("phase_offset").at("phase_rad").get<double>();
            channel.add<PhaseOffsetImpairment>(phase_rad);
        }
        else if (s == "timing_offset") {
            const int sample_offset = simConfig.at("imp").at("timing_offset").at("sample_offset").get<int>();
            const std::string mode_str = simConfig.at("imp").at("timing_offset").at("mode").get<std::string>();
            TimingOffsetImpairment::Mode mode = TimingOffsetImpairment::Mode::Circular;
            if (mode_str == "ZeroPad") { mode = TimingOffsetImpairment::Mode::ZeroPad; }
            channel.add<TimingOffsetImpairment>(sample_offset, mode);
        }
        else if (s == "awgn") {
            const double snr_db = simConfig.at("imp").at("awgn").at("snr_db").get<double>();
            channel.add<AWGNImpairment>(snr_db);
        }
    }
    logger.log(Logger::Level::INFO, "Channel initialized.");

    DataWriter writer("dumps", "basic_tx");
    logger.log(Logger::Level::INFO, "DataWriter initialized at " + writer.run_dir().string());

    for (uint64_t iter = 0; iter < frameCount; iter++) {
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
