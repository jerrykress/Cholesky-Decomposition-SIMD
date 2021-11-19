#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <immintrin.h> // AVX
#include <iomanip>
#include <chrono>
#include <stdint.h>
#include <bits/stdc++.h>
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

// Input Buffer
vector<vector<float>> matrixVec;
// Matrix Storage
float **matrix_f32;
double **matrix_f64;
// Dimension
int dim;

/** 
    Horizontal sum reduce of float vector 
**/
inline float hsum_float_avx(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
    vlow = _mm_add_ps(vlow, vhigh);             // reduce down to 128

    __m128 high64 = _mm_unpackhi_ps(vlow, vlow);
    return _mm_cvtss_f32(_mm_add_ss(vlow, high64)); // reduce to scalar
}

/** 
    Horizontal sum reduce of double vector 
**/
inline double hsum_double_avx(__m256d v)
{
    __m128d vlow = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
    vlow = _mm_add_pd(vlow, vhigh);              // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); // reduce to scalar
}

/**
    Cholesky Float 32
**/
float **cholesky_f32(float **L, int n)
{
    int i, j, k, num_f32x8;

    __m256 q, p = _mm256_set1_ps(0);

    for (j = 0; j < n; j++)
    {
        memset(&L[j][j + 1], 0, sizeof(float) * (n - j - 1));
        // printf("$1\n");
        num_f32x8 = j / 8;

        for (k = 0; k < num_f32x8; k++)
        {
            // printf("k=%d\n", k);
            q = _mm256_loadu_ps(&L[j][k * 8]);
            // printf("$1a\n");
            q = _mm256_mul_ps(q, q);
            // printf("$1b\n");
            L[j][j] -= hsum_float_avx(q);
            // printf("$1c\n");
        }
        // printf("$2\n");

        for (k = num_f32x8 * 8; k < j; k++)
        {
            L[j][j] = fma(-L[j][k], L[j][k], L[j][j]);
        }
        // printf("$3\n");

        L[j][j] = sqrt(L[j][j]);
        // printf("$4\n");

        for (i = j + 1; i < n; i++)
        {
            for (k = 0; k < num_f32x8; k++)
            {
                q = _mm256_loadu_ps(&L[i][k * 8]);
                p = _mm256_loadu_ps(&L[j][k * 8]);
                p = _mm256_mul_ps(q, p);
                L[i][j] -= hsum_float_avx(p);
            }

            for (k = num_f32x8 * 8; k < j; k++)
            {
                L[i][j] = fma(-L[i][k], L[j][k], L[i][j]);
            }

            L[i][j] /= L[j][j];
        }
        // printf("$5\n");
    }

    return L;
}

/**
    Cholesky Float 64
**/
double **cholesky_f64(double **L, int n)
{
    int i, j, k, num_f64x4;

    __m256d q, p = _mm256_set1_pd(0);

    for (j = 0; j < n; j++)
    {
        memset(&L[j][j + 1], 0, sizeof(double) * (n - j - 1));
        // printf("$1\n");
        num_f64x4 = j / 4;

        for (k = 0; k < num_f64x4; k++)
        {
            // printf("k=%d\n", k);
            q = _mm256_loadu_pd(&L[j][k * 4]);
            // printf("$1a\n");
            q = _mm256_mul_pd(q, q);
            // printf("$1b\n");
            L[j][j] -= hsum_double_avx(q);
            // printf("$1c\n");
        }
        // printf("$2\n");

        for (k = num_f64x4 * 4; k < j; k++)
        {
            L[j][j] = fma(-L[j][k], L[j][k], L[j][j]);
        }
        // printf("$3\n");

        L[j][j] = sqrt(L[j][j]);
        // printf("$4\n");

        for (i = j + 1; i < n; i++)
        {
            for (k = 0; k < num_f64x4; k++)
            {
                q = _mm256_loadu_pd(&L[i][k * 4]);
                p = _mm256_loadu_pd(&L[j][k * 4]);
                p = _mm256_mul_pd(q, p);
                L[i][j] -= hsum_double_avx(p);
            }

            for (k = num_f64x4 * 4; k < j; k++)
            {
                L[i][j] = fma(-L[i][k], L[j][k], L[i][j]);
            }

            L[i][j] /= L[j][j];
        }
        // printf("$5\n");
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
