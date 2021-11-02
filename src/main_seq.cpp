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

    for (j = 0; j < n; j++)
    {
        // Replace with 0
        memset(&L[j][j + 1], 0, sizeof(T) * (n - j - 1));
        i = j;

        // Diagnal

        for (k = 0; k < i; k++)
        {
            L[j][j] -= L[j][k] * L[j][k];
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

    // check if file exists
    if (!file.good())
    {
        cout << "Error! Such file does not exist: " << filename << endl;
        return;
    }

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
            // cout << "[READ] " << substr << endl;
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

/*
    Write matrix output to a file
*/
template <class T>
void writeOuput(T **L)
{
    string fn_out = "./output/main_seq_out_" + to_string(dim) + ".txt";
    ofstream output(fn_out, std::ofstream::trunc);
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            output << setprecision(5) << L[i][j] << " ";
        }
        output << endl;
    }
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

    // cout << "[INFO] DIM=" << dim << endl;

    /* Performing a timed task */
    // printMatrix(matrix, dim, "Original Matrix");

    auto t1 = high_resolution_clock::now();
    matrix = cholesky<float>(matrix, dim);
    auto t2 = high_resolution_clock::now();

    // printMatrix(matrix, dim, "Decomposed Matrix");
    writeOuput<float>(matrix);

    /* Getting number of milliseconds as a double */
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "\033[32m[RESULT] " << ms_double.count() << " ms\033[0m"
              << endl;

    return 0;
}
