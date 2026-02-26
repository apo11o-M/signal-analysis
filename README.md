# Signal Analysis

This is a "end-to-end receiver" signal analysis project implemented in C++. It's a learning project for signal processing and communication systems from the perspective of a computer science person without much knowledge about electronics ¯\\\_(ツ)_/¯

## Project Structure

This project parses a waveform json file, which then generates signals and transmits them through a simulated channel with added impairments (noise, frequency offset, timing offset, etc), see the `tx/` directory for more details. The received signals are then processed to recover the original data, see `rx/` for more info. Visualization is done through the `visual/` module where it contains python scripts to plot various logging data from both stages of the program.

- `core/`: Core signal processing components and utilities
- `tx/`: Transmitter implementations
- `rx/`: Receiver implementations
- `visual/`: Visualization tools and modules
- `utils/`: Utility functions and helpers that doesn't fit into `core/`

## Build & Run

There are two stages of the program, running the simulation, and visualizing the results. One can run both stages separately or use the `run.py` script to run both stages in one go.

```bash
# run the default simulation and visualization
cd scripts
python run.py
```

Or, if you want to run the stages separately, you can do:

```bash
# build and run the simulation
mkdir build
cd build
cmake ..
cmake --build .

# run the simulation with a config file
./signal-analysis.exe ../sim_config/chirp.json

# visualize the results
cd ../visual
python visual plot_simulation.py
```

## Useful Books and Resources

- `Signals, Systems and Transforms` by Leland B Jackson
- `Digital Communications, Fundamentals and Applications 2nd Edition` by Bernard Sklar
