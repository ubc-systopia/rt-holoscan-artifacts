# Artifact 

## Overview

Our artifact has two components. The first is a GitHub repository with code to replicate
the majority of the results in our paper (figures 8-11). The repository also contains a PDF of the paper itself.

The second component is two Docker containers with the necessary dependencies to replicate the remaining UPPAAL graph (figure 7). This README has instructions for the repository experiments, while a separate README in the uppaal-results directory of the repository has instructions specific to the containers.

## Platform Details

**Submitted Paper Evaluation Platform.** 
For the paper, we ran all experiments using an Ubuntu 22.04.5 Linux server with a 64 core Intel Xeon Gold 6326 CPU. Execution time of specific Holoscan apps was profiled using a
NVIDIA [IGX Orin](https://docs.nvidia.com/igx/) Developer Kit with a 12-core Arm Cortex-A78AE CPU
64 GB RAM, and an NVIDIA RTX 6000 Ada discrete GPU.

**Recommendations.** To run the repository experiments, we recommend using a server with as much RAM as 
possible (the server we used has 1000G of RAM), as memory requirements become intensive when running our algorithm with large problem instances. If on a machine with little available RAM, it may not be possible to replicate the full results in the paper due to a lack of available memory (e.g., it may be possible to get results for 25 node graphs, but not 30 node graphs).

### Docker containers

To obtain worst-case execution times for Holoscan applications and use them to generate and run the UPPAAL models, consult the README in the uppaal-results directory.

An arm64 machine with an NVIDIA GPU is necessary to run the first part of the experiment (WCET profiling).

If you do not have access to such a machine, you can use the pre-generated WCETs and run only the UPPAAL models using the commands listed in the README.

## Repository Setup

The sole requirement for the repository experiments is an 
installation of the [Python](https://www.python.org/downloads/release/python-3120/) 
programming language (we specifically used Python 3.12 and reproduced using Python 3.10; similar versions will likely also work).

The repository can cloned with the following command: 

    git clone https://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@github.com/pschowitz/rtss2025-artifact.git

Once the repository is cloned, navigate to it:

    cd rtss2025-artifact

Finally, install the required Python packages with the below command.

    pip install -r requirements.txt

## Running experiments

### master.py

The script `master.py` reproduces our experimental results using the files in the `code` folder. The experiments fall into three categories, described here. More information on all output can be found in the `Output` section.

The scalability experiment gradually runs our TG-DFS algorithm on randomly generated problem instances
of increasing complexity, tracking how the runtime, expression count, and proportion of time spent in different steps varies. Creates `runtime_boxplots.pdf`, `runtime_line.pdf`, and `exprcount_boxplots.pdf`.

The queue scaling experiment uses a fixed set of known graphs and measures how the runtime of TG_DFS increases as we run the algorithm with increasing queue sizes. Creates `qsizescaling.pdf`.

The prior work pessimism experiment evaluates the pessimism of the results presented in Schowitz 2024 [1]. Creates `priorpessimism.txt`.

`master.py` has the following arguments:

- `--all` or `-a`
    - Option to run all experiments.
- `--scalability` or `-s`
    - Option to run scalability experiment.
- `--queuesize` or `-q`
    - Option to run queue scaling experiment.
- `--priorpessimism` or `-p`
    - Option to run prior work pessimism experiment.
- `--nodemax`
    - The maximum amount of nodes in synthetic graphs, must be multiple of 5. Default value is 20, 30 is used in the paper.
- `--queuemax`
    - The maximum queue size for queue scaling experiments. Default value is 20, 40 is used in the paper.
- `--trials`
    - The number of trials to run in the prior work pessimism experiments. Default value is 25, 100 is used in the paper.

The scalability experiments cover a user-defined range of graphs, as determined by `nodemax` and `queuemax`. 

The runtime of the experiments is largely dependent on the values of `nodemax` and `queuemax`, as execution times and memory requirements will scale exponentially as they are increased. There is also
significant variation within one class of graph, e.g., a graph with 30 nodes may take the average 
time of a graph with 35 nodes, or a graph with 25 nodes, which can lead to long runtimes and out of memory errors at high node counts or queue sizes due to the exponential scaling. The runtime of the prior work pessimism experiments scales linearly with `trials`.

#### Run all experiments

To run all the experiments with default settings (shorter experiments than in the paper), use `--all`.
We measured total experiment runtime under this configuration as ~6 minutes on our server.

    python master.py --all

To come as close as possible to replicating the full results shown in the paper, use the following command. Please note that the full scalability and queuesize experiments may take several hours to complete and were run overnight for the paper. The lengthy runtime is largely taken up by the execution of a few of the largest graphs, as can be seen in figures 9 and 11, and more extreme outliers may also occur. If the experiment seems to have stalled, it is likely a graph which will take multiple hours to finish processing has been randomly generated.

    python master.py --all --nodemax 30 --queuemax 40 --trials 100

When the experiments finish, the `data` folder will be populated with all the resulting raw data and figures, described in the `Output` section below.

## Output

After running `master.py` with the `-a` option, the `data` folder will contain the following files, with the first five corresponding directly to results from the paper:

- exprcount_boxplots.pdf
    - Graph showing node count vs expression count for synthetic graphs. Corresponds to figure 8 in the paper.
- runtime_boxplots.pdf  
    - Graph showing node count vs runtime for synthetic graphs. Corresponds to figure 9 in the paper. 
- runtime_line.pdf  
    - Graph showing node count versus proportion of runtime spent in TG-DFS and maximisation for synthetic graphs. Corresponds to figure 10 in the paper.
- qsizescaling.pdf  
    - Graph showing queue size versus runtime for four Holoscan graphs. Corresponds to figure 11 in the paper.
- priorpessimism.txt
    - Text file showing average pessimism for eight Holoscan applications using the analysis technique in Schowitz 2024 [1]. Corresponds to table II in the paper.
- exprcount.csv  
    - List of expression counts for synthetic graphs. Used to create exprcount_boxplots.pdf.
- scalability.csv  
    - List of runtimes for synthetic graphs. Used to create runtime_boxplots.pdf and runtime_line.pdf.
- qsizescaling.csv  
    - List of runtimes for four Holoscan graphs. Used to create qsizescaling.pdf.


## Code and inputs

The following documents the files in the `code` directory and other inputs used by `master.py` to populate `data`.

### code/TG_DFS.py

Our algorithm for finding the worst-case response time. Takes a DOT file describing 
a graph along with execution times, and computes an exact bound on the WCRT
under our system model. For the purposes of these experiments, it returns runtimes
and the count of how many possible expressions were maximized over to find
the WCRT.

### code/experiments.py

Uses the synthetic graphs in `dot_outputs` created by `createdot.py` to evaluate runtime and
expression count of our TG-DFS algorithm by calling to `TG_DFS.py`. Uses the Holoscan graph structures in `holoscan_dot` to run queue scaling experiments. Also uses `computeWCRT.py` and `runsimulation.py`
to compare the WCRTs given by the analysis in Schowitz 2024 to simulated results over many sets of 
execution times for the `holoscan_dot` graphs [1].

Generates three files: `exprcount.csv` `scalability.csv` and `qsizescaling.csv`. 

### code/plotfromcsv.py

Plotting script for scalability and queue scaling experiments.

### code/createdot.py

This file creates random DOT files and is used to generate synthetic graphs
for the scalability experiment. Also generates execution times for operators
in the graph. Each synthetic graph is saved to a directory `synthetic_dot`
with a name indicating how many nodes it has. For example `G_10_2.dot` is the 
second generated graph with 10 nodes. These are newly generated each time the scalability 
experiment runs.

### holoscan_dot

A directory with DOT files encoding graph structures of 8 Holoscan applications used in the queue
scaling experiment and prior pessimism experiments. These are the 8 graphs shown in figure 7.

### code/computeWCRT.py and code/runsimulation.py

These are taken from the artifact of the paper that we motivate
our results against. Find the artifact and citation below.

Artifact link:

    https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/holoscan_response_time_analysis/artifact

[1] P. Schowitz, S. Sinha, and A. Gujarati, “Response-Time Analysis 
of a Soft Real-time NVIDIA Holoscan Application,” in IEEE Real-Time 
Systems Symposium, 2024.