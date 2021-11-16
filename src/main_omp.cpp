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
#include <omp.h>

using namespace std;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// Input Buffer
vector<vector<float>> matrixVec;
// Matrix Storage
float **matrix_f32;
double **matrix_f64;
// Dimension
int dim;

/**
    Cholesky Float 32
**/

float **cholesky_f32(float **L, int n)
{
    cout << "omp_get_num_procs: " << omp_get_num_procs() << endl;
    cout << "omp_get_max_threads: " << omp_get_max_threads() << endl;

    int i, j, num_f32x4x4;
    int batchSize = 0;

    float32x4_t lane = vdupq_n_f32(0);
    float32x4x4_t q4, p4;

    for (j = 0; j < n; j++)
    {
        memset(&L[j][j + 1], 0, sizeof(float) * (n - j - 1));
        i = j;

        num_f32x4x4 = i / 16;

        for (int k = 0; k < num_f32x4x4; k++)
        {
            printf("$1 Thread: %d, i=%d, j=%d, k=%d\n", omp_get_thread_num(), i, j, k);
            lane = vdupq_n_f32(0);
            q4 = vld4q_f32(&L[j][k * 16]);
            lane = vmlsq_f32(lane, q4.val[0], q4.val[0]);
            lane = vmlsq_f32(lane, q4.val[1], q4.val[1]);
            lane = vmlsq_f32(lane, q4.val[2], q4.val[2]);
            lane = vmlsq_f32(lane, q4.val[3], q4.val[3]);
            printf("$2 Thread: %d, k=%d\n", omp_get_thread_num(), k);
            L[j][j] += vaddvq_f32(lane);
            printf("$3 Thread: %d, k=%d\n", omp_get_thread_num(), k);
        }

        for (int k = num_f32x4x4 * 16; k < i; k++)
        {
            L[j][j] = fma(-L[j][k], L[j][k], L[j][j]);
        }

        L[j][j] = sqrt(L[j][j]);

        for (i = j + 1; i < n; i++)
        {
            num_f32x4x4 = j / 16;

            for (int k = 0; k < num_f32x4x4; k++)
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

            for (int k = num_f32x4x4 * 16; k < j; k++)
            {
                L[i][j] = fma(-L[i][k], L[j][k], L[i][j]);
            }

            L[i][j] /= L[j][j];
        }
    }

    return L;
}

/**
    Cholesky Float 64
**/
double **cholesky_f64(double **L, int n)
{
    int i, j, k, num_f64x2x4;
    int batchSize = 0;

    float64x2_t lane = vdupq_n_f64(0);
    float64x2x4_t q4, p4;

    for (j = 0; j < n; j++)
    {
        memset(&L[j][j + 1], 0, sizeof(double) * (n - j - 1));
        i = j;

        num_f64x2x4 = i / 8;

        for (k = 0; k < num_f64x2x4; k++)
        {
            lane = vdupq_n_f64(0);
            q4 = vld4q_f64(&L[j][k * 8]);
            lane = vmlsq_f64(lane, q4.val[0], q4.val[0]);
            lane = vmlsq_f64(lane, q4.val[1], q4.val[1]);
            lane = vmlsq_f64(lane, q4.val[2], q4.val[2]);
            lane = vmlsq_f64(lane, q4.val[3], q4.val[3]);
            L[j][j] += vaddvq_f64(lane);
        }

        for (k = num_f64x2x4 * 8; k < i; k++)
        {
            L[j][j] = fma(-L[j][k], L[j][k], L[j][j]);
        }

        L[j][j] = sqrt(L[j][j]);

        for (i = j + 1; i < n; i++)
        {
            num_f64x2x4 = j / 8;

            for (k = 0; k < num_f64x2x4; k++)
            {
                lane = vdupq_n_f64(0);
                q4 = vld4q_f64(&L[i][k * 8]);
                p4 = vld4q_f64(&L[j][k * 8]);
                lane = vmlsq_f64(lane, q4.val[0], p4.val[0]);
                lane = vmlsq_f64(lane, q4.val[1], p4.val[1]);
                lane = vmlsq_f64(lane, q4.val[2], p4.val[2]);
                lane = vmlsq_f64(lane, q4.val[3], p4.val[3]);
                L[i][j] += vaddvq_f64(lane);
            }

            for (k = num_f64x2x4 * 8; k < j; k++)
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
template <class T>
T **initMatrix(T **destination, vector<vector<float>> &origin, int n)
{
    destination = new T *[n];

    for (int i = 0; i < n; i++)
    {
        destination[i] = new T[n];
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
void writeOuput(T **L, double duration)
{
    // write the decomposed matrix
    string fn_out = "./output/main_neon_out_" + to_string(dim) + ".txt";
    ofstream matrix_output(fn_out, std::ofstream::trunc);
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            matrix_output << setprecision(6) << L[i][j] << " ";
        }
        matrix_output << endl;
    }

    // write performance information
    string fn_perf = "./output/main_neon_perf_" + to_string(dim) + ".txt";
    ofstream perf_output(fn_perf, std::ofstream::trunc);
    perf_output << duration << endl;
}

int main(int argc, char **argv)
{
    /* Read filename */
    string filename = argv[1];
    /* Read Type of Data = 1,2,3 */
    int test_mode = stoi(argv[2]);
    /* Timer */
    high_resolution_clock::time_point t1, t2;
    /* Read matrix into vector buffer */
    readMatrix(filename, matrixVec);

    /* If Input valid, run test */
    if (validMatrix(matrixVec))
    {
        duration<double, std::milli> ms_double;
        switch (test_mode)
        {
        case 1:
            // float32
            matrix_f32 = initMatrix<float>(matrix_f32, matrixVec, dim);
            t1 = high_resolution_clock::now();
            matrix_f32 = cholesky_f32(matrix_f32, dim);
            t2 = high_resolution_clock::now();
            ms_double = t2 - t1;
            std::cout
                << "\033[32m[RESULT] <f32> " << ms_double.count() << " ms\033[0m"
                << endl;
            writeOuput<float>(matrix_f32, ms_double.count());
            break;

        case 2:
            // float64
            matrix_f64 = initMatrix<double>(matrix_f64, matrixVec, dim);
            t1 = high_resolution_clock::now();
            matrix_f64 = cholesky_f64(matrix_f64, dim);
            t2 = high_resolution_clock::now();
            ms_double = t2 - t1;
            std::cout
                << "\033[32m[RESULT] <f64> " << ms_double.count() << " ms\033[0m"
                << endl;
            writeOuput<double>(matrix_f64, ms_double.count());
            break;

        default:
            break;
        }
    }

    return 0;
}
