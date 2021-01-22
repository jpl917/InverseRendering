#ifndef HEADERS_H
#define HEADERS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SVD>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
//#include <Eigen/CholmodSupport>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <Eigen/LU>
#include <Eigen/StdVector>

#include "TriMesh.h"
#include "TriMesh_algo.h"

using namespace std;

struct float2
{
    float x, y;
};

struct float3
{
    float x, y, z;
    float3():x(0.), y(0.), z(0.){}
    float3(float _x, float _y, float _z): x(_x), y(_y), z(_z){}
};


#endif
