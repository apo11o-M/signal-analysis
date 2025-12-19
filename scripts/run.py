# Python script that runs the signal analysis application
# can handle compile, execute, and visualization tasks

from pathlib import Path
import argparse, subprocess, sys

CURR_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT_DIR / "build"
IS_WINDOWS = sys.platform.startswith("win")

def run(cmd, cwd=None):
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    print("=====================================================================")
    print("[INFO] Signal analysis runner starting..")
    print("=====================================================================")
    
    arg_parser = argparse.ArgumentParser(description="Run the signal analysis application")
    arg_parser.add_argument("--build", default=True, help="compile project")
    arg_parser.add_argument("--run", default=True, help="execute the application")
    arg_parser.add_argument("--viz", default=True, help="visualize results")

    args = arg_parser.parse_args()
    
    if args.build:
        BUILD_DIR.mkdir(exist_ok=True)
        run(["cmake", "--build", str(BUILD_DIR)])

    if args.run:
        if IS_WINDOWS:
            run([str(BUILD_DIR / "Debug" / "signal-analysis.exe")])
        else:
            run([str(BUILD_DIR / "signal-analysis")])

    if args.viz:
        # get latest dump directory
        dump_dir = sorted((CURR_DIR / "dumps").iterdir())[-1]
        
        run(["python", str(ROOT_DIR / "visual" / "plot_tx_log.py"), "--run", \
            str(dump_dir), "--stage", "tx", "--frame", "0", "--fs", "1e7", "--center"])

    print("=====================================================================")    
    print("[INFO] Signal analysis runner finished")
    print("=====================================================================")

if __name__ == "__main__":
    main()
