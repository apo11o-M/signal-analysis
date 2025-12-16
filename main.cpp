#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Map {0,1} -> {+1,-1}
static inline float bit_to_pm1(int b) { return b ? -1.0f : +1.0f; }

// Simple maximal-length LFSR m-sequence generator.
// This is not "the" PRN for any standard; it's a compact demo.
struct LFSR {
    // state must be non-zero
    uint16_t state;
    // taps as bitmask (feedback XOR of selected bits). You can tweak.
    uint16_t taps;

    explicit LFSR(uint16_t seed = 0xACE1u, uint16_t tapsMask = 0xB400u)
        : state(seed ? seed : 1u), taps(tapsMask) {}

    // returns next chip in {+1,-1}
    float next_chip_pm1() {
        // Galois LFSR step (common form)
        uint16_t lsb = state & 1u;
        state >>= 1;
        if (lsb) state ^= taps;
        // use output bit (lsb) as pseudo-chip
        return lsb ? -1.0f : +1.0f;
    }
};

struct TxParams {
    int numBits = 200;          // payload bits
    int prnLen = 1023;          // chips per bit (spreading factor)
    int samplesPerChip = 8;     // rectangular pulse shaping
    uint32_t rngSeed = 12345;
};

int main(int argc, char** argv) {
    TxParams p;
    if (argc >= 2) p.numBits = std::stoi(argv[1]);
    if (argc >= 3) p.prnLen = std::stoi(argv[2]);
    if (argc >= 4) p.samplesPerChip = std::stoi(argv[3]);

    if (p.numBits <= 0 || p.prnLen <= 0 || p.samplesPerChip <= 0) {
        std::cerr << "Invalid args. Usage: tx_demo [numBits] [prnLen] [samplesPerChip]\n";
        return 1;
    }

    // Random bits
    std::mt19937 rng(p.rngSeed);
    std::uniform_int_distribution<int> bitDist(0, 1);
    std::vector<int> bits(p.numBits);
    for (int i = 0; i < p.numBits; ++i) bits[i] = bitDist(rng);

    // PRN generator
    LFSR prn(/*seed=*/0xACE1u, /*taps=*/0xB400u);

    // Generate IQ samples
    // Total chips = numBits * prnLen
    // Total samples = totalChips * samplesPerChip
    const int64_t totalChips = static_cast<int64_t>(p.numBits) * p.prnLen;
    const int64_t totalSamples = totalChips * p.samplesPerChip;

    std::ofstream iq("tx_iq.csv");
    iq << "n,i,q\n";
    iq << std::fixed << std::setprecision(6);

    int64_t n = 0;
    for (int b = 0; b < p.numBits; ++b) {
        const float bitVal = bit_to_pm1(bits[b]);
        for (int c = 0; c < p.prnLen; ++c) {
            const float chip = prn.next_chip_pm1();
            const float symbolChip = bitVal * chip; // spread chip
            // rectangular pulse shaping: repeat samplesPerChip times
            for (int s = 0; s < p.samplesPerChip; ++s) {
                iq << n << "," << symbolChip << "," << 0.0f << "\n";
                ++n;
            }
        }
    }
    iq.close();

    std::ofstream meta("tx_meta.txt");
    meta << "numBits=" << p.numBits << "\n";
    meta << "prnLen=" << p.prnLen << "\n";
    meta << "samplesPerChip=" << p.samplesPerChip << "\n";
    meta << "totalSamples=" << totalSamples << "\n";
    meta << "rngSeed=" << p.rngSeed << "\n";
    meta.close();

    std::cout << "Wrote tx_iq.csv and tx_meta.txt\n";
    std::cout << "Args used: numBits=" << p.numBits
              << " prnLen=" << p.prnLen
              << " samplesPerChip=" << p.samplesPerChip << "\n";

    return 0;
}

// std::cout << "Program Starts.." << std::endl;
// std::cout << "..Finished" << std::endl;
