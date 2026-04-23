# Assignment 5 - Write a Fast Histogram Kernel

**Due May 5, 11:59pm. There are no late days for this assignment.**

## Overview

In this assignment, you will optimize a multi-channel histogram kernel and make it run as fast as possible. Treat this as an open-ended mini final project: there is no single correct implementation strategy, and there is no fixed performance bar for full credit.

You may implement the kernel in CUDA, Triton, TileLang, PyTorch, or a combination of these tools. The starter code includes a CUDA example so you can see how Python calls into a `.cu` file directly, but you are not required to use that path in your final solution.

## Repository setup.
Create a private copy of the release repository:

- Go to GitHub import page: <https://github.com/new/import>
- Import from: <https://github.com/princeton-ece476/assignment5>
- Set the new repository to **Private**
- Add your partner in GitHub repository settings under: *Settings → Collaborators*

Then clone your repository and add the release repo as a remote:

```shell
git clone git@github.com:<your-github-name>/<your-repo-name>.git
cd <your-repo-name>
git remote add release https://github.com/princeton-ece476/assignment3
```

## Environment Setup

Use the `adroit-vis.princeton.edu` head node for this assignment.

First-time environment setup:
```sh
# Load envinronment modules and the srun helper
source a5_profile.sh

# Make Python environment. 
conda create -n ece476_assignment5 python==3.12
conda activate ece476_assignment5
pip install -r ./requirements.txt
```

Then, every time you begin a new session:
```sh
# Load envinronment modules and the srun helper
source a5_profile.sh

# Load Python environment. 
conda activate ece476_assignment5
```

## Files
- `histogram.py`: Entrypoint. It provides the `test`, `benchmark`, and `profile` commands. You should not need to edit this file.
- `student_kernel.py`: Student entry point. This is the file you will be working on.
- `histogram.cu`: If you implemented with CUDA, put your kernel here. `student_kernel.py` contains starter code that compiles and loads this moudle.
- `reference.py`: the correctness and runtime reference.
- `test_cases/test.txt`: the default benchmark shape.

## Task: Multi-Channel Histogram

Implement `custom_kernel(data)` in `student_kernel.py`.

`data` is `(array, num_bins)`:

- `array`: CUDA tensor with shape `[length, num_channels]`, dtype `torch.uint8`
- `num_bins`: number of bins, with values in `array` in `[0, num_bins - 1]`

Return a `torch.int32` tensor with shape `[num_channels, num_bins]`, where `histogram[c, b]` is the count of value `b` in channel `c`.

The starter `student_kernel.py` calls into `histogram.cu` using PyTorch's C++/CUDA extension loader. You may keep that path, replace the CUDA kernel, or implement `custom_kernel` with Triton, TileLang, or another local approach, as long as custom_kernel retains its current function signature.

`histogram.py` preloads `student_kernel.preload_custom_kernel()` before timing so extension compilation is not included in the reported kernel runtime. If your implementation needs setup before timing, put it in `preload_custom_kernel()`. If no setup is needed, it can be an empty function.

### Test Cases
The default test case is: `length: 1048576; num_channels: 512; num_bins: 256; seed: 1001`

To use a different case, create a text file in the same key/value format and pass it as the second argument:

```bash
python histogram.py benchmark my_cases.txt
```

The autograder will only do a mimimal correctness checking - The majority of this assignment will be your work log.

### Running, Testing and Benchmarking

For fast correctness checks, you can run on the head node directly. Make sure you are on the adroit-vis headnode.
```sh
python histogram.py test        # Short Test
```

For reliable results, submit to compute nodes.
```sh
a5-srun python histogram.py test        # Short Test
a5-srun python histogram.py benchmark   # Runs benchmark and compare with reference.
```

### Profiling
To have a deeper understanding of your code's performance and bottlenecks, you need to use a profiler.

Profiling can only be done on the compute nodes. Permissions aren't granted to users on the head node to read performance counters.
```sh
a5-srun python histogram.py profile
```
This writes an `.ncu-rep` file under `profile_data/`. This file is best viewed with a UI. There are many ways to do this.
- Go to https://myadroit.princeton.edu/, in "Interative Apps", start "Desktop" and connect to it. Start a terminal and navigate to your assignment directory.
- Alternatively, if you know how to setup X-Forwarding, you can start from there.
- Alternatively, you can install [Nsight Compute on your computer](https://developer.nvidia.com/tools-overview/nsight-compute/get-started), download the `histogram_profile.ncu-rep` to your computer, and open it there.

Run `ncu-ui profile_data/histogram_profile.ncu-rep`. Alternatively, run `ncu-ui` which starts the program. In Files -> Open File, find your `ncu-rep` file.

![ncu-ui](images/Screenshot%202026-04-22%20212049.png)

Go through the different kernels (select from the "Result" dropdown) and different tabs to find useful information on how to improve.

## What To Submit

Submit on GradeScope. 
- **Programming Assignment 5 (Worklog):** Submit a PDF file detailing your optimization procedure. For each iteration, it should include the following:
    1. How the code was structured at that step.
    2. The measured runtime.
    3. The profiling or measurement data you used.
    4. What you concluded from the measurements.
    5. What hypothesis guided the next change.
    6. Why you stopped optimizing.
- **Programming Assignment 5 (Code):** Submit a ZIP archive of your GitHub repository. Download ZIP from GitHub or choose your repository on GradeScope.
    - You may include intermediate versions described in your worklog.

## Grading

This assignment is graded on a combination of effort, measurement-driven reasoning, and final performance:

- **80 points:** minimal but meaningful optimization effort, with a work log showing course concepts and some speedup.
- **95 points:** solid optimization effort, clear interpretation of runtimes/profiles, and a reasonable sequence of design choices.
- **95-110 points:** exceptional effort and/or unusually strong final performance, awarded case by case.

The staff may assign lower scores if the submitted work does not demonstrate meaningful engagement with the optimization process.
