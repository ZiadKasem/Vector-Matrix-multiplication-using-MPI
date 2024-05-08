# MPI-based Dense Matrix-Vector Multiplication

## Description
This repository contains an MPI-based program written in C for multiplying a dense n x n matrix A with a vector B in parallel. The program follows a parallel algorithm where each process computes different portions of the resulting product vector.

## Features
- Parallel computation using MPI.
- Efficient distribution of matrix A and vector B across processes.
- Supports both Matrix-Vector and Vector-Matrix multiplication based on user input.

## Usage
1. Clone the repository to your local machine:

```bash
git clone <repository_url>
```

2. Compile the program using an MPI compiler (e.g., `mpicc`):

```bash
mpicc -o matrix_vector_multiplication matrix_vector_multiplication.c
```

3. Run the compiled executable with the desired number of MPI processes:

```bash
mpiexec -n <num_processes> ./matrix_vector_multiplication
```

4. Follow the prompts to enter the matrix size and elements, as well as the vector elements.

5. Choose between Matrix-Vector or Vector-Matrix multiplication.

6. View the result printed to the console.

## Example
Here is an example of running the program with 4 MPI processes:

```bash
mpiexec -n 4 ./matrix_vector_multiplication
```

## Dependencies
- MPI library
- C compiler (e.g., GCC)

## Contributors
- [Ziad Ashraf Ashraf]([https://github.com/your_username](https://github.com/ZiadKasem))


Feel free to contribute to the project by submitting pull requests or reporting issues. If you have any questions or suggestions, please contact [Ziad Ashraf Ashraf]([https://github.com/your_username](https://github.com/ZiadKasem)).
