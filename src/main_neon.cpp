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
#include <unistd.h>

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
    int i, j, k, num_f32x4x4;
    int batchSize = 0;
    float batchSum = 0;

    float32x4_t lane = vdupq_n_f32(0);
    float32x4_t q1, p1;
    float32x4x2_t q2, p2;
    float32x4x3_t q3, p3;
    float32x4x4_t q4, p4;

    for (j = 0; j < n; j++)
    {
        memset(&L[j][j + 1], 0, sizeof(T) * (n - j - 1));
        i = j;

        num_f32x4x4 = i / 16;

        for (k = 0; k < num_f32x4x4; k++)
        {
            lane = vdupq_n_f32(0);
            q4 = vld4q_f32(&L[j][k * 16]);
            lane = vmlsq_f32(lane, q4.val[0], q4.val[0]);
            lane = vmlsq_f32(lane, q4.val[1], q4.val[1]);
            lane = vmlsq_f32(lane, q4.val[2], q4.val[2]);
            lane = vmlsq_f32(lane, q4.val[3], q4.val[3]);
            L[j][j] += vaddvq_f32(lane);
        }

        for (k = num_f32x4x4 * 16; k < i; k++)
        {
            L[j][j] = fma(-L[j][k], L[j][k], L[j][j]);
        }

        L[j][j] = sqrt(L[j][j]);

        for (i = j + 1; i < n; i++)
        {
            num_f32x4x4 = j / 16;

            for (k = 0; k < num_f32x4x4; k++)
            {
                lane = vdupq_n_f32(0);
                q4 = vld4q_f32(&L[i][k * 16]);
                p4 = vld4q_f32(&L[j][k * 16]);
                lane = vmlsq_f32(lane, q4.val[0], p4.val[0]);
                lane = vmlsq_f32(lane, q4.val[1], p4.val[1]);
                lane = vmlsq_f32(lane, q4.val[2], p4.val[2]);
                lane = vmlsq_f32(lane, q4.val[3], p4.val[3]);
                L[i][j] += vaddvq_f32(lane);
            }

            for (k = num_f32x4x4 * 16; k < j; k++)
            {
                L[i][j] = fma(-L[i][k], L[j][k], L[i][j]);
            }

            L[i][j] /= L[j][j];
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
            // cout << "[INIT]" << destination[i][j] << endl;
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
    string fn_out = "./output/main_neon_out_" + to_string(dim) + ".txt";
    ofstream output(fn_out, std::ofstream::trunc);
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            output << L[i][j] << " ";
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
