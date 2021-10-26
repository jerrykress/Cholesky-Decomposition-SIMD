#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdint.h>
#include <bits/stdc++.h>
#include <arm_neon.h>
#include <math.h>
#include <fstream>
#include <string>
#include <stdexcept>
#include <sstream>

using namespace std;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// Global Variables
vector<vector<float>> matrixVec;
float **matrix;
int dim;

/**
    Perform Int cholesky decomposition and return result
**/
template <class T>
T **cholesky(T **L, int n)
{
    int i, j, k;
    float32x4_t q1, q2, prod;
    float batchSum = 0;

    for (j = 0; j < n; j++)
    {
        // Replace with 0
        memset(&L[j][j + 1], 0, sizeof(T) * (n - j - 1));
        i = j;

        // Diagnal NEON
        for (k = 0; k + 3 < i; k += 4)
        {
            q1 = vld1q_f32(&L[j][k]);
            prod = vmulq_f32(q1, q1);
            batchSum = vaddvq_f32(prod);
            L[j][j] -= batchSum;
        }
        // Diagnal Remainder
        for (k = k; k < i; k++)
        {
            L[j][j] -= L[j][k] * L[j][k];
        }

        L[i][i] = sqrt(L[j][j]);

        for (i = j + 1; i < n; i++)
        {
            // Bottom NEON
            for (k = 0; k + 3 < j; k += 4)
            {
                q1 = vld1q_f32(&L[i][k]);
                q2 = vld1q_f32(&L[j][k]);
                prod = vmulq_f32(q1, q2);
                batchSum = vaddvq_f32(prod);
                L[i][j] -= batchSum;
            }
            // Bottom Remainder
            for (k = k; k < j; k++)
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
void printMatrix(float **L, int n, string text = "Print Matrix")
{
    cout << "[PRINT] " << text << endl;
    cout << "=================" << endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << setprecision(2) << L[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

/**
    Initialise the matrix 2d array from vector
**/
float **initMatrix(float **destination, vector<vector<float>> &origin, int n)
{
    destination = new float *[n];

    for (int i = 0; i < n; i++)
    {
        destination[i] = new float[n];
        for (int j = 0; j < n; j++)
        {
            destination[i][j] = origin[i][j];
            cout << "[INIT]" << destination[i][j] << endl;
        }
    }
    return destination;
}

void readMatrix(string filename, vector<vector<float>> &destination, char delim = ' ')
{
    // if destination vector not empty, abort
    if (!destination.empty())
    {
        cout << "Destination matrix not empty. Abort..." << endl;
        return;
    }

    // setup stream and line buffer
    ifstream file(filename);
    string line;

    // read file line by line
    while (getline(file, line))
    {
        // prepare line buffer
        vector<float> v;
        stringstream ss(line);

        // parse line using delimiter
        while (ss.good())
        {
            string substr;
            getline(ss, substr, delim);
            cout << "[READ] " << substr << endl;
            v.push_back(stof(substr));
        }

        // add line to destination matrix
        destination.push_back(v);
    }
}

/**
 * Check if a matrix is square and valid
**/
bool validMatrix(vector<vector<float>> &m)
{
    int len = m.size();

    for (int i = 0; i < len; i++)
    {
        if (m[i].size() != len)
        {
            cout << "Error! Matrix shape incorrect." << endl;
            return false;
        }
    }

    dim = len;
    return true;
}

int main(int argc, char **argv)
{
    string filename = argv[1];
    readMatrix(filename, matrixVec);

    /* Initialise */
    if (validMatrix(matrixVec))
    {
        matrix = initMatrix(matrix, matrixVec, dim);
    }

    cout << "[INFO] DIM=" << dim << endl;

    /* Performing a timed task */
    printMatrix(matrix, dim, "Original Matrix");

    auto t1 = high_resolution_clock::now();
    matrix = cholesky<float>(matrix, dim);
    auto t2 = high_resolution_clock::now();

    printMatrix(matrix, dim, "Decomposed Matrix");

    /* Getting number of milliseconds as a double */
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "[RESULT] " << ms_double.count() << " ms\n"
              << endl;

    return 0;
}
