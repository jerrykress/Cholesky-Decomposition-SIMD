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

#define MAX 100

using namespace std;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

//TODO: Calculate how many number of iterations are needed
//TODO: Get array of 8*8 or similar number sets and multiply accumulate, deduct from the result
//TODO: In the case where a number set is not completly filled. we need to verify if the remaining positions are 0 so they won't affect the results

// Global Variables
vector<vector<double>> matrixVec;
double **matrix;
int dim;

/**
    Perform Int cholesky decomposition and return result
**/
double **cholesky(double **L, int n)
{
    int i, j, k;
    int num8x16 = ceil(n / 16);

    for (j = 0; j < n; j++)
    {
        // Replace with 0
        memset(&L[j][j + 1], 0, sizeof(double) * (n - j - 1));
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
void printMatrix(double **L, int n, string text = "Print Matrix")
{
    cout << text << endl;
    cout << "=================" << endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << setprecision(2) << L[i][j] << " ";
            // printf("%f\n", L[i][j]);
        }
        cout << endl;
    }
    cout << endl;
}

/**
    Initialise the matrix 2d array from vector
**/
double **initMatrix(double **destination, vector<vector<double>> origin, int n)
{
    destination = new double *[n];

    for (int i = 0; i < n; i++)
    {
        destination[i] = new double[n];
        for (int j = 0; j < n; j++)
        {
            destination[i][j] = origin[i][j];
            cout << "[INIT]" << destination[i][j] << endl;
        }
    }
    return destination;
}

void readMatrix(string filename, vector<vector<double>> &destination, char delim = ' ')
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
        vector<double> v;
        stringstream ss(line);

        // parse line using delimiter
        while (ss.good())
        {
            string substr;
            getline(ss, substr, delim);
            v.push_back(stod(substr));
        }

        // add line to destination matrix
        destination.push_back(v);
    }
}

/**
 * Check if a matrix is square and valid
**/
bool validMatrix(vector<vector<double>> &m)
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
    // string filename = argv[0];
    string filename = "input_16x16.txt";
    readMatrix(filename, matrixVec);

    /* Initialise */
    if (validMatrix(matrixVec))
    {
        matrix = initMatrix(matrix, matrixVec, dim);
    }

    cout << "[INFO] DIM=" << dim << endl;

    /* Performing a timed task */
    auto t1 = high_resolution_clock::now();
    printMatrix(matrix, dim, "Original Matrix");
    matrix = cholesky(matrix, dim);
    printMatrix(matrix, dim, "Decomposed Matrix");
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double */
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << " ms" << endl;

    return 0;
}
