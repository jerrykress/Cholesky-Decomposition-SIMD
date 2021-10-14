#include <bits/stdc++.h>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "arm_neon.h"
#include <stdint.h>

#define MAX 100

using namespace std;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

/**
    Perform Int cholesky decomposition and return result
**/
int **cholesky(int **L, int n)
{
    int i, j, k;

    for (j = 0; j < n; j++)
    {

        for (i = 0; i < j; i++)
        {
            L[i][j] = 0;
        }

        for (k = 0; k < i; k++)
        {
            L[j][j] = L[j][j] - L[j][k] * L[j][k];
        }

        L[i][i] = sqrt(L[j][j]);

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

    return 0;
}
