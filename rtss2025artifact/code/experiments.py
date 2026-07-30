import argparse
import os
import re
import random
from code.createdot import generate_dag_dot
import code.TG_DFS
import code.computeWCRT
import code.runsimulation

def schowitz2024(trials):
    rf = open("data/priorpessimism.txt", 'w')

    for i in  ["a", "b", "c", "d", "e", "f", "g", "h"]: 
        filepath = "holoscan_dot/graph" + i + ".dot"

        running_counterSIM = 0
        running_counterWCRT = 0

        for j in range(trials):
            randomize_wcet(filepath, 100, 1000)

            running_counterWCRT += code.computeWCRT.main(filepath)
            running_counterSIM += code.runsimulation.main(filepath, 1000000, 1, 1, False)

        rf.write("Average percent pessimism for Graph " + i + ": " + str(((running_counterWCRT-running_counterSIM)/running_counterSIM)*100) + "\n")
    rf.close()


def randomize_wcet(dot_file, min_val, max_val):
    with open(dot_file, "r") as f:
        content = f.read()

    content = re.sub(r'(\[WCET=)(\d+)(\])', lambda m: f"{m.group(1)}{random.randint(min_val, max_val)}{m.group(3)}", content)

    with open(dot_file, "w") as f:
        f.write(content)

def run_qsizescaling(max):
    rf = open("data/qsizescaling.csv", 'w')
    for i in ["e", "f", "g", "h"]:
        filepath = "holoscan_dot/graph" + i + ".dot"

        WCRTs = []
        for queuesize in range(max):
            randomize_wcet(filepath, 100, 1000)
            # The last value returned is expr count, unnecessary for this experiment
            WCRTs.append(code.TG_DFS.main(filepath, queuesize, True)[:-1])

        rf.write(f"{WCRTs}\n")
    rf.close()

def run_scalability(start_n=5, end_n=20, step=5, runs=10, min_wcet=100, 
                   max_wcet=1000, extra_edge_prob=.2, graph_name='G', out_dir='synthetic_dot'):
    os.makedirs(out_dir, exist_ok=True)

    with open("data/scalability.csv", 'w') as rf:
        rf.write("#nodes,runID,time1,time2,q2time1,q2time2,q3time1,q3time2,\n")
    rf = open("data/scalability.csv", 'a')

    results = []
    Q2_results = []
    Q3_results = []

    with open("data/exprcount.csv", 'w') as rf2:
        rf2.write("#nodes,runID,exprcount,exprcountQ2,exprcountQ3\n")
    rf2 = open("data/exprcount.csv", 'a')

    expr_results = []
    exprQ2_results = []
    exprQ3_results = []

    for n in range(start_n, end_n + 1, step):
        for r in range(1, runs + 1):
            dot_file = os.path.join(out_dir, f"{graph_name}_{n}_{r}.dot")
            generate_dag_dot(n, min_wcet, max_wcet, extra_edge_prob, graph_name, output=dot_file)
            # Run tree traversal and record time
            t1, t2, exprs = code.TG_DFS.main(dot_file, 1, True)
            q2t1, q2t2, exprsQ2 = code.TG_DFS.main(dot_file, 2, True)
            q3t1, q3t2, exprsQ3 = code.TG_DFS.main(dot_file, 3, True)
            # Append to dynamic log
            rf.write(f"{n},{r},{t1},{t2},{q2t1},{q2t2},{q3t1},{q3t2}\n")
            rf.flush()
            rf2.write(f"{n},{r},{exprs},{exprsQ2},{exprsQ3}\n")
            rf.flush()
            # Store for plotting
            results.append((n, t1+t2))
            Q2_results.append((n, q2t1+q2t2))
            Q3_results.append((n, q3t1+q3t2))
            expr_results.append((n, exprs))
            exprQ2_results.append((n, exprsQ2))
            exprQ3_results.append((n, exprsQ3))

    rf.close()
    rf2.close()


def scalability_main():
    parser = argparse.ArgumentParser(description="Run DAG traversal experiments over varying node counts.")
    parser.add_argument('--start-n', type=int, default=5,
                        help='Starting number of nodes')
    parser.add_argument('--end-n', type=int, default=20,
                        help='Ending number of nodes')
    parser.add_argument('--step', type=int, default=5,
                        help='Step size for node count')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs per node count')
    parser.add_argument('--min-wcet', type=int, default=100,
                        help='Minimum WCET for nodes')
    parser.add_argument('--max-wcet', type=int, default=1000,
                        help='Maximum WCET for nodes')
    parser.add_argument('-p', '--prob', type=float, default=0.2,
                        help='Probability of adding an extra edge')
    parser.add_argument('-g', '--graph-name', type=str, default='G',
                        help='Prefix for graph file names')
    parser.add_argument('-o', '--out-dir', type=str, default='synthetic_dot',
                        help='Output directory for dot files and results')


    args = parser.parse_args()
    run_scalability(
        args.start_n,
        args.end_n,
        args.step,
        args.runs,
        args.min_wcet,
        args.max_wcet,
        args.prob,
        args.graph_name,
        args.out_dir,
    )

if __name__ == '__main__':
    scalability_main()
