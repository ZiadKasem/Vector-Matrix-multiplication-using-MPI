#pragma warning(disable:4996) // Disables the warning for using unsafe functions in Visual Studio
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

#define MAX_SIZE 100 // Maximum value for size

int main(int argc, char** argv) {
    int rank, size, N_SizeOfMatrix;
    int data[MAX_SIZE * MAX_SIZE]; // Matrix data
    int vector[MAX_SIZE]; // Vector data
    int result[MAX_SIZE]; // Result of matrix-vector multiplication
    int Vector_Matrix = 1;

    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    // Process 0 prompts user to enter matrix size
    if (rank == 0) {
        do {
            printf("Enter 0 for Matrix-Vector Multiplication or 1 for Vector-Matrix multiplication: ");
            fflush(stdout);
            scanf("%d", &Vector_Matrix);
        } while (Vector_Matrix != 0 && Vector_Matrix != 1);
        

        printf("Enter the size of the square matrix (max %d): ", MAX_SIZE);
        fflush(stdout);
        scanf("%d", &N_SizeOfMatrix);
        if (N_SizeOfMatrix > MAX_SIZE) {
            printf("Error: Matrix size exceeds maximum allowed size %d.\n", MAX_SIZE);
            fflush(stdout);
            MPI_Finalize();
            return 1;
        }

        // Input matrix elements
        printf("Enter the elements of the matrix:\n");
        for (int i = 0; i < N_SizeOfMatrix; i++) {
            for (int j = 0; j < N_SizeOfMatrix; j++) {
                printf("Enter element (%d,%d): ", i, j);
                fflush(stdout);
                scanf("%d", &data[i * N_SizeOfMatrix + j]);
            }
        }

        // Input Vector elements
        printf("Enter the elements of the Vector:\n");
        for (int i = 0; i < N_SizeOfMatrix; i++) {
            printf("Enter element (%d): ", i);
            fflush(stdout);
            scanf("%d", &vector[i]);
        }

        // For Vector Matrix Multiplication apply transposition
        if (Vector_Matrix == 1) {
            int temp;
            for (int i = 0; i < N_SizeOfMatrix; ++i) {
                for (int j = i + 1; j < N_SizeOfMatrix; ++j) {
                    // Swap elements data[i][j] and data[j][i]
                    temp = data[i * N_SizeOfMatrix + j];
                    data[i * N_SizeOfMatrix + j] = data[j * N_SizeOfMatrix + i];
                    data[j * N_SizeOfMatrix + i] = temp;
                }
            }
        }
    }

    // Broadcast matrix size to all processes
    MPI_Bcast(&N_SizeOfMatrix, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast the vector to all processes
    MPI_Bcast(vector, N_SizeOfMatrix, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the chunk size for each process
    int chunk_size = N_SizeOfMatrix / size;

    // If the number of processes is greater than or equal to the matrix size
    if (size >= N_SizeOfMatrix) {
        chunk_size = 1;

        // Allocate memory for receiving a chunk of data matrix
        int recv_chunk[MAX_SIZE * MAX_SIZE];

        // Scatter the data matrix among all processes
        MPI_Scatter(data, chunk_size * N_SizeOfMatrix, MPI_INT, recv_chunk,
            chunk_size * N_SizeOfMatrix, MPI_INT, 0, MPI_COMM_WORLD);

        // Perform local computation: matrix-vector multiplication
        for (int i = 0; i < chunk_size; i++) {
            result[i] = 0;
            for (int j = 0; j < N_SizeOfMatrix; j++) {
                result[i] += recv_chunk[i * N_SizeOfMatrix + j] * vector[j];
            }
        }

        // Gather the results from all processes
        MPI_Gather(result, chunk_size, MPI_INT, recv_chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (Vector_Matrix == 1) {
                printf("Result of  Vector_Matrix multiplication is size 1X%d:\n", N_SizeOfMatrix);
                for (int i = 0; i < N_SizeOfMatrix; i++) {
                    printf("%d\n", recv_chunk[i]);
                }
            }
            else {
                printf("Result of  Matrix-Vector multiplication is size %dX1:\n", N_SizeOfMatrix);
                for (int i = 0; i < N_SizeOfMatrix; i++) {
                    printf("%d ", recv_chunk[i]);
                }
            }


        }
    }
    else {
        // Calculate the chunk size for each process
        int chunk_size = N_SizeOfMatrix / size;
        int remainder = N_SizeOfMatrix % size;

        // Allocate memory for receiving a chunk of data matrix
        int recv_chunk[MAX_SIZE * MAX_SIZE];

        // Scatter the data matrix among all processes
        MPI_Scatter(data, chunk_size * N_SizeOfMatrix, MPI_INT, recv_chunk,
            chunk_size * N_SizeOfMatrix, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate the actual chunk size for this process
        int actual_chunk_size = (rank < remainder) ? chunk_size + 1 : chunk_size;

        // Perform local computation: matrix-vector multiplication
        for (int i = 0; i < actual_chunk_size; i++) {
            result[i] = 0;
            for (int j = 0; j < N_SizeOfMatrix; j++) {
                result[i] += recv_chunk[i * N_SizeOfMatrix + j] * vector[j];
            }
        }

        // Gather the results from all processes
        int recv_counts[MAX_SIZE];
        int displs[MAX_SIZE];
        int total_recv_count = 0;

        MPI_Gather(&actual_chunk_size, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            displs[0] = 0;
            total_recv_count = recv_counts[0];
            for (int i = 1; i < size; i++) {
                total_recv_count += recv_counts[i];
                displs[i] = displs[i - 1] + recv_counts[i - 1];
            }
        }

        MPI_Gatherv(result, actual_chunk_size, MPI_INT, recv_chunk, recv_counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (Vector_Matrix == 1) {
                printf("Result of Vector_Matrix multiplication is size 1X%d:\n", N_SizeOfMatrix);
                for (int i = 0; i < N_SizeOfMatrix; i++) {
                    printf("%d\n", recv_chunk[i]);
                }
            }
            else {
                printf("Result of Matrix-Vector multiplication is size %dX1:\n", N_SizeOfMatrix);
                for (int i = 0; i < N_SizeOfMatrix; i++) {
                    printf("%d ", recv_chunk[i]);
                }
            }
        }
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}
