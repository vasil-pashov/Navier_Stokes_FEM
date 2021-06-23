import argparse
from collections import defaultdict
import subprocess
import re
import statistics

def parseArgs():
    parser = argparse.ArgumentParser(description='Measure the performace of fem_soler')
    parser.add_argument("exePath", metavar="<Path to executable>", help="Path to fem_solver executable which is going to be meassured.")
    parser.add_argument("sceneFile", metavar="<Path to scene>", help="Path to the fem_solver scene description.")
    parser.add_argument("-r", "--runs", help="How many times to repeat the execution of the solver.", default=21, type=int)
    parser.add_argument("-t", "--numThreads", help="Number of threads which the solver will use. Default - all threads available.", type=int)
    return parser.parse_args()

def parseOutput(otput, allFunctionTimes):
    pattern = re.compile("\[Scoped Timer\]\[(.*)\] ([-+]?\d*\.\d+|\d+)s")
    allMatches = re.findall(pattern, otput)
    for (fn, time) in allMatches:
        print("{}: {}".format(fn, time))
        allFunctionTimes[fn].append(float(time))

def processTimings(allFunctionTimes, totalRuns):
    print("========================================================================")
    print("Stats for all timers")
    print("Number of runs: {}\n".format(totalRuns))
    for (fn, timings) in allFunctionTimes.items():
        if(len(timings) != totalRuns):
            print("[WARNING] This function was run {} times. The requested amout was {}".format(len(timings), totalRuns))
        print("Function: {}".format(fn))
        print("Mean: {}".format(statistics.fmean(timings)))
        print("Median: {}".format(statistics.median_high(timings)))
        print("SD: {}".format(statistics.stdev(timings)))
        print("Min: {}".format(min(timings)))
        print("Max: {}".format(max(timings)))

def main():
    args = parseArgs()
    exeArgs = ["-sceneFile={}".format(args.sceneFile)]
    if args.numThreads is not None:
        exeArgs.append("-numThreads={}".format(args.numThreads))
    command = [args.exePath] + exeArgs
    allFunctionTimes = defaultdict(lambda: [])
    for i in range(0, args.runs):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Run {} completed".format(i))
            parseOutput(result.stdout, allFunctionTimes)
        except subprocess.CalledProcessError as ex:
            print("Error: {} occured while trying to run running {}".format(result.stderr, command))
            raise ex
    processTimings(allFunctionTimes, args.runs)

if __name__ == "__main__":
    main()