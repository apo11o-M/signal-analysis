#pragma once

#include "json.hpp"
#include "Logger.hpp"

#include <iostream>
#include <fstream>
#include <unordered_map>

using json = nlohmann::json;

class SimConfig {
public:
    SimConfig(const std::string& filename) : filename_(filename) {
        std::ifstream file(filename);
        config_ = json::parse(file);
    }

    // some quick format check to see if the json does contain the required
    // elements. this does not fully guarantee the json config is entirely valid
    // for all transmitters, impairments, and receivers
    bool validateJson(Logger& logger) {
        bool valid = true;
        if (!config_.contains("tx")) {
            logger.log(Logger::Level::ERROR, "Config json does not contain the required 'tx' element");
            valid = false;
        }
        if (!config_.contains("imp")) {
            logger.log(Logger::Level::ERROR, "Config json does not contain the required 'imp' element");
            valid = false;
        }
        if (!config_.contains("rx")) {
            logger.log(Logger::Level::ERROR, "Config json does not contain the required 'rx' element");
            valid = false;
        }

        if (!config_.contains("frame_count")) {
            logger.log(Logger::Level::ERROR, "Config json does not contain the required 'frame_count' element");
            valid = false;
        }

        if (!config_.contains("frame_size")) {
            logger.log(Logger::Level::ERROR, "Config json does not contain the required 'frame_size' element");
            valid = false;
        }

        return valid;
    }

    json& at(const std::string& str) {
        return config_.at(str);
    }

    const json& at(const std::string& str) const {
        return config_.at(str);
    }

private:
    std::string filename_;
    json config_;

};
