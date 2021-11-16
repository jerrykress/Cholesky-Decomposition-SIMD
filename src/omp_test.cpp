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
#include <omp.h>

using namespace std;

void test_loop()
{
    int j = 0;
    int s = 0;

    cout << "omp_get_num_procs: " << omp_get_num_procs() << endl;
    cout << "omp_get_max_threads: " << omp_get_max_threads() << endl;

    for (int k = 0; k < 2; k++)
    {
#pragma omp parallel for
        for (int i = 0; i < 10; i++)
        {
            j = omp_get_thread_num();
            j *= 2;
            s = j;
            printf("Thread: %d/%d, i=%d, s=%d\n", omp_get_thread_num(), omp_get_num_threads(), i, s);
        }
        cout << "====================" << endl;
    }
}

int main()
{
    test_loop();
    return 0;
}