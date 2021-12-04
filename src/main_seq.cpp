#include "matrix_utils.h"

using namespace std;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// Global Variables
vector<vector<float>> matrixVec;
float **matrix_f32;
double **matrix_f64;
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

    /* Initialise */
    if (validMatrix(matrixVec, dim))
    {
        duration<double, std::milli> ms_double;

        switch (test_mode)
        {
        case 1:
            // float 32
            matrix_f32 = initMatrix<float>(matrix_f32, matrixVec, dim);
            t1 = high_resolution_clock::now();
            matrix_f32 = cholesky<float>(matrix_f32, dim);
            t2 = high_resolution_clock::now();
            ms_double = t2 - t1;
            std::cout
                << "\033[32m[RESULT] <f32> " << ms_double.count() << " ms\033[0m"
                << endl;
            writeOuput<float>(matrix_f32, dim, ms_double.count(), "main_seq_out_", "main_seq_perf_");
            break;

        case 2:
            // float 64
            matrix_f64 = initMatrix<double>(matrix_f64, matrixVec, dim);
            t1 = high_resolution_clock::now();
            matrix_f64 = cholesky<double>(matrix_f64, dim);
            t2 = high_resolution_clock::now();
            ms_double = t2 - t1;
            std::cout
                << "\033[32m[RESULT] <f64> " << ms_double.count() << " ms\033[0m"
                << endl;
            writeOuput<double>(matrix_f64, dim, ms_double.count(), "main_seq_out_", "main_seq_perf_");
            break;

        default:
            break;
        }
    }

    return 0;
}
