#include "matrix_utils.h"

#include <arm_neon.h>
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
    int i, j, k, num_f32x4x4;
    float ljj = 0;

    float32x4_t lane = vdupq_n_f32(0);
    float32x4x4_t q4, p4;

    for (j = 0; j < n; j++)
    {
        num_f32x4x4 = j / 16;

        memset(&L[j][j + 1], 0, sizeof(float) * (n - j - 1));

#pragma omp parallel for reduction(+ \
                                   : ljj) private(q4, p4, lane)
        for (k = 0; k < num_f32x4x4; k++)
        {
            lane = vdupq_n_f32(0);
            q4 = vld4q_f32(&L[j][k * 16]);
            lane = vmlsq_f32(lane, q4.val[0], q4.val[0]);
            lane = vmlsq_f32(lane, q4.val[1], q4.val[1]);
            lane = vmlsq_f32(lane, q4.val[2], q4.val[2]);
            lane = vmlsq_f32(lane, q4.val[3], q4.val[3]);
            ljj += vaddvq_f32(lane);
        }

#pragma omp parallel for reduction(- \
                                   : ljj)
        for (k = num_f32x4x4 * 16; k < j; k++)
        {
            ljj -= L[j][k] * L[j][k];
        }

        L[j][j] += ljj;
        L[j][j] = sqrt(L[j][j]);
        ljj = 0;

#pragma omp parallel for private(q4, p4, lane, k)
        for (i = j + 1; i < n; i++)
        {
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
    Cholesky Float 64
**/
double **cholesky_f64(double **L, int n)
{
    int i, j, k, num_f64x2x4;
    double ljj = 0;

    float64x2_t lane = vdupq_n_f64(0);
    float64x2x4_t q4, p4;

    for (j = 0; j < n; j++)
    {
        num_f64x2x4 = j / 8;

        memset(&L[j][j + 1], 0, sizeof(double) * (n - j - 1));

#pragma omp parallel for reduction(+ \
                                   : ljj) private(q4, p4, lane)
        for (k = 0; k < num_f64x2x4; k++)
        {
            lane = vdupq_n_f64(0);
            q4 = vld4q_f64(&L[j][k * 8]);
            lane = vmlsq_f64(lane, q4.val[0], q4.val[0]);
            lane = vmlsq_f64(lane, q4.val[1], q4.val[1]);
            lane = vmlsq_f64(lane, q4.val[2], q4.val[2]);
            lane = vmlsq_f64(lane, q4.val[3], q4.val[3]);
            ljj += vaddvq_f64(lane);
        }

#pragma omp parallel for reduction(- \
                                   : ljj)
        for (k = num_f64x2x4 * 8; k < j; k++)
        {
            ljj -= L[j][k] * L[j][k];
        }

        L[j][j] += ljj;
        L[j][j] = sqrt(L[j][j]);
        ljj = 0;

#pragma omp parallel for private(q4, p4, lane, k)
        for (i = j + 1; i < n; i++)
        {
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
    if (validMatrix(matrixVec, dim))
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
            writeOuput<float>(matrix_f32, dim, ms_double.count(), "main_simd_out_", "main_simd_perf_");
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
            writeOuput<double>(matrix_f64, dim, ms_double.count(), "main_simd_out_", "main_simd_perf_");
            break;

        default:
            break;
        }
    }

    return 0;
}
