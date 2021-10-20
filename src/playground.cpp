#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include <stdint.h>
#include <bits/stdc++.h>
#include <arm_neon.h>
#include <math.h>

#define MAX 100

using namespace std;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

//TODO: Calculate how many number of iterations are needed
//TODO: Get array of 8*8 or similar number sets and multiply accumulate, deduct from the result
//TODO: In the case where a number set is not completly filled. we need to verify if the remaining positions are 0 so they won't affect the results

/**
    Perform Int cholesky decomposition and return result
**/
int **cholesky(int **L, int n)
{
    int i, j, k;

    for (j = 0; j < n; j++)
    {
        // Replace with 0
        memset(&L[j][j + 1], 0, sizeof(int) * (n - j - 1));
        i = j;

        // Diagnal
        for (k = 0; k < i; k++)
        {
            L[j][j] = L[j][j] - L[j][k] * L[j][k];
        }
        L[i][i] = sqrt(L[j][j]);

        // Calculate left
        for (i = j + 1; i < n; i++)
        {
            for (k = 0; k < j; k++)
            {
                L[i][j] = L[i][j] - L[i][k] * L[j][k];
            }
            L[i][j] = L[i][j] / L[j][j];
        }
    }

    return L;
}

/**
    Print a matrix
**/
void printMatrix(int **L, int n, string text = "Print Matrix")
{
    cout << text << endl;
    cout << "=================" << endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << L[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

/**
    Initialise the matrix from array
**/
template <size_t N, size_t M>
int **initMatrix(int **matrix, int (&raw)[N][M], int n)
{
    matrix = new int *[n];

    for (int i = 0; i < n; i++)
    {
        matrix[i] = new int[n];
        for (int j = 0; j < n; j++)
        {
            matrix[i][j] = raw[i][j];
            // cout << "Init Pos[" << i << "][" << j << "]: " << matrix[i][j] << endl;
        }
    }
    return matrix;
}

int main()
{
    int **matrix;

    int raw[][MAX] = {{4, 12, -16},
                      {12, 37, -43},
                      {-16, -43, 98}};

    /* Finding the size of the matrix */
    int n = sizeof(raw) / sizeof(raw[0]);

    /* Initialise */
    matrix = initMatrix(matrix, raw, n);

    /* Performing a timed task */
    auto t1 = high_resolution_clock::now();
    printMatrix(matrix, n, "Original Matrix");
    matrix = cholesky(matrix, n);
    printMatrix(matrix, n, "Decomposed Matrix");
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double */
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << " ms" << endl;

    /* more tests */
    uint8x16_t neonArr1 = {0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0};

    uint8x16_t neonArr2 = {0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0};

    uint8_t *arr1 = new uint8_t[12];
    arr1[1] = 1;
    arr1[2] = 2;

    uint8_t *arr2 = new uint8_t[12];
    arr2[1] = 3;
    arr2[2] = 4;

    neonArr1 = vld1q_u8(arr1);
    neonArr2 = vld1q_u8(arr2);
    neonArr2 = vmulq_u8(neonArr1, neonArr2);

    // vst1q_u8(arr, neonArr2);
    uint8_t sum = vaddvq_u8(neonArr2);

    // for (int i = 0; i < 16; i++)
    // {
    //     cout << unsigned(arr[i]) << ", " << endl;
    // }

    cout << endl
         << "Sum: " << unsigned(sum) << endl;

    return 0;
}
