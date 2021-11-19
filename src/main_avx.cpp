#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <immintrin.h>

using namespace std;

int main()
{
    __m256 evens = _mm256_set_ps(2, 4, 6, 8, 10, 12, 14, 16);
    __m256 odds = _mm256_set_ps(1, 3, 5, 7, 9, 11, 13, 15);

    __m256 result = _mm256_sub_ps(evens, odds);

    float *f = (float *)&result;

    for (int i = 0; i < 8; i++)
    {
        cout << f[i] << " ";
    }
    cout << endl;

    return 0;
}