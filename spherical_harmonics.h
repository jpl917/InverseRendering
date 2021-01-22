#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H

//http://silviojemma.com/public/papers/lighting/spherical-harmonic-lighting.pdf

#include <iostream>
#include <vector>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "global.h"

using namespace std;

double clamp(double c, double lower = 0.0, double upper = 1.0);

// 1*2*3*...*n
double factorial(int n);

double K(int l, int m);

double P(int l,int m,double x);

double SH(int m, int l, double phi, double theta);

// phi: azimuth angle
// theta: polar angle
void sh_basis(double phi, double theta, std::vector<double>& coeff, const int& order=3);


void cart2sph(double nx, double ny, double nz, double& phi, double& theta);


void cal_SH_from_envmap(const std::string& filename, std::vector<std::vector<double> >& sh_coeff, const int& order = 3);


#endif
