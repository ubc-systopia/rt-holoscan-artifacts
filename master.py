import argparse
import os
from code.experiments import run_scalability, run_qsizescaling, schowitz2024
from code.plotfromcsv import plot_scalabilityboxplots_from_csv, plot_scalabilityline_from_csv, plot_exprcount_from_csv, plot_qsizescaling
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", action="store_true", help="Enable all experiments")
    parser.add_argument("-s", "--scalability", action="store_true", help="Enable scalability experiments")
    parser.add_argument("-q", "--queuesize", action="store_true", help="Enable queue scaling experiments")
    parser.add_argument("-p", "--priorpessimism", action="store_true", help="Enable prior work pessimism experiments")
    parser.add_argument(
        "--nodemax",
        type=int,
        help="The maximum number of nodes for scalability experiments. Default = 20",
        default=20,
    )
    parser.add_argument(
        "--queuemax",
        type=int,
        help="The maximum queue size for queue scaling experiments. Default = 20",
        default=20,
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="The number of trials for the prior. Default = 50",
        default=25,
    )

    os.makedirs('data', exist_ok=True)

    args = parser.parse_args()

    if args.scalability or args.all:
        print("Running scalability experiments...")
        run_scalability(end_n = args.nodemax)
        plot_scalabilityline_from_csv()
        plot_scalabilityboxplots_from_csv()
        plot_exprcount_from_csv()
        print("Scalability experiments complete")

    if args.queuesize or args.all:
        print("Running queuesize experiments...")
        run_qsizescaling(args.queuemax)
        plot_qsizescaling()
        print("Queuesize experiments complete")

    if args.priorpessimism or args.all:
        print("Running prior pessimism experiments...")
        schowitz2024(args.trials)
        print("Prior pessimism experiments complete")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Runtime: {end - start:.4f} seconds")