# Layer VQE

This repository contains the code, experiments, and analysis developed for the study of Layer Variational Quantum Eigensolvers (L-VQE) and their application to combinatorial optimization problems. Its scope is to reproduce the results from [1] and go beyond it with other benchmarking experiments.
The primary goal of this project is to evaluate the performance of the L-VQE algorithm and compare it against established variational quantum algorithms, namely Variational Quantum Eigensolvers (VQE) and the Quantum Approximate Optimization Algorithm (QAOA).

The benchmarking experiments focus on two representative optimization problems: k-Community Detection and MaxCut. For each problem, we analyze solution quality, optimization behavior, the impact of circuit entanglement, and robustness under realistic noise models. Where possible, simulations are complemented by executions on IBM Quantum hardware to assess the practical viability of the proposed approach.

Due to the significant computational cost of the experiments, intermediate and final results were stored in CSV files and later used to generate the figures, plots, and tables presented in the accompanying report and presentation.

This document describes the repository structure, explains the purpose of each experiment, and provides instructions for reproducing the reported results.

## Structure
- folder [experiments](experiments) : due to issues with using functions from different folders, we decided to have all ipynb notebooks with each main experiment in this file
   - [lvqe_engine](experiments/l_vqe_engine.py): complete L-VQE framework, including Hamiltonian construction, ansatz generation, optimization routines, and execution engines for L-VQE, VQE, and QAOA algorithms. It supports both simulation and real quantum hardware execution (IBM Quantum).
   - K-community detection notebooks:
      - [lvqe_vs_qaoa](experiments/01_Community_Detection_Analysis_lvqe_vs_qaoa.ipynb)
      - [entanglement vs no entanglement](experiments/01_entagnlement_Community_Detection_Analysis.ipynb)
      - [lvqe_vs_vqe](experiments/01_lvqe_vs_vqe_Community_Detection_Analysis.ipynb)
      - [noisy lvqe vs vqe on aer simulator](experiments/Community_Detection_qiskit_noisy.ipynb)
      - [ibm harrdware lvqe short run](experiments/simulate_noisy_k_comm.py) -> main focus was not on this simulation due to the limited time of 10 minutes access per month given by IBM. However, if given an credentials token, this simulation can be used to slightly compare the results of the Qiskit Aer noise simulation. Due to hardware and time constraints, we can say that we cannot fully rely on the results of the noisy simulations. 

   - MaxCut notebooks:
      - [lvqe_vs_qaoa](experiments/02_MaxCut_Analysis_lvqe_vs_qaoa.ipynb)
      - [entanglement_vs_no-entanglement](experiments/02_entagnlement_MaxCut_Analysis.ipynb)
      - [lvqe_vs_vqe](experiments/02_lvqe_vs_vqe_MaxCut_Analysis.ipynb)
      - [noisy lvqe vs vqe on aer simulator](experiments/MaxCut_noisy.ipynb) -> way better results due to simplicity of the problem.


- folder [plots](plots)
   - contains figures and plots used in the presentation. all of them were generated in the experiments presented above.

- folder [csv_files](csv_files)
   - due to long run times, the experiments's results were saved in the csv files.
   - these results were used to create the plots and tables used in the presentation.


## Future improvements 
   - better noisy comparison: use 1000-2000 shots in the Qiskit aer simulation to create a better comparison between L-VQE and VQE and also make comparison with real hardware. Get full access to real hardware to be able to better compare the results.
   - entanglement vs no entanglement experiments: replace CNOT gates with T gates randomly; go beyond paper.
   - run algorithm against other combinatorial problems such as the Traveling salesman probem.

   ## Software Stack

The implementation relies primarily on:

- PennyLane
- Qiskit
- Qiskit Aer
- NumPy
- SciPy
- NetworkX
- Matplotlib

Experiments were developed and tested using Python 3.12.



## Getting Started

### Prerequisites
- Python 3.12 ([download here](https://www.python.org/downloads/))

### Installation

1. Clone the repository

2. Create a virtual environment
```bash
   python3.12 -m venv venv
```

3. Activate the virtual environment
```bash
   # macOS/Linux
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
```

4. Install dependencies
```bash
   pip install -r requirements.txt
```

## References

[1] X. Liu, A. Angone, R. Shaydulin, I. Safro, Y. Alexeev, and L. Cincio,
"Layer VQE: A Variational Approach for Combinatorial Optimization on Noisy Quantum Computers,"
*IEEE Transactions on Quantum Engineering*, vol. 3, pp. 1–15, 2022.

## Use of AI

AI was used in order to understand certain concepts and approaches of understanding the above presented concepts. It was also used for code refactoring, certain parameters choosing and better understanding the documentation of certain libraries that were used.

