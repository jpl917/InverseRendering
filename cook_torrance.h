#ifndef COOK_TORRANCE_H
#define COOK_TORRANCE_H

#include <iostream>
#include <cmath>
#include <Eigen/Core>

#define PI 3.1415926535

double D_GGX(double NoH, double roughness);

double D_BlinnPhong(double NoH, double roughness);

double _G1(double NoM, double k);

double G_Smith(double NoV, double NoL, double roughness);

double G_min(double NoH, double NoV, double NoL, double VoH);

double Fresnel(double F0, double VoH);

Eigen::Vector3d disney(Eigen::Vector3d l, Eigen::Vector3d v, 
                       Eigen::Vector3d n, Eigen::Vector3d d, 
                       double r, double s);

#endif
