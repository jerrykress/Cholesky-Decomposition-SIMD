#include <stdio.h>
#include <assert.h>
#include <iostream>
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
bool validMatrix(vector<vector<float>> &m, int &dim)
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
void writeOuput(T **L, int dim, double duration, string matrix_fn, string perf_fn)
{
    // write the decomposed matrix
    string fn_out = "./output/" + matrix_fn + to_string(dim) + ".txt";
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
    string fn_perf = "./output/" + perf_fn + to_string(dim) + ".txt";
    ofstream perf_output(fn_perf, std::ofstream::trunc);
    perf_output << duration << endl;
}