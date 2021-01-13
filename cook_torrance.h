#ifndef COOK_TORRANCE_H
#define COOK_TORRANCE_H

//

#include <iostream>
#include <cmath>
#include <Eigen/Core>

#define PI 3.1415926535

double maximum(double x, double y){
    return x > y ? x : y ;
}

double D_GGX(double NoH, double roughness){
    double alpha = roughness * roughness;
    double tmp = alpha / maximum(1e-8, (NoH * NoH * (alpha * alpha - 1.0) + 1.0) );
    return tmp * tmp / PI;
}

// double D_Bechmann(double NoH, double roughness){
//     double alpha = roughness * roughness;
//     double r1 = 1.0 / (alpha * pow(NoH, 4));
//     double r2 = (NoH * NoH - 1.0) / (alpha * NoH * NoH);
//     return r1 * exp(r2);
// }

double D_BlinnPhong(double NoH, double roughness){
    return roughness * pow(NoH, 12) + (1.0 - roughness) * pow(NoH, 48);
}


double _G1(double NoM, double k){
    return NoM / (NoM * (1.0 - k) + k);
}

double G_Smith(double NoV, double NoL, double roughness){
    double k = maximum(1e-8, roughness * roughness / 2.0);
    //double k = maximum(1e-8, (roughness + 1) * (roughness + 1) / 8.0);
    return _G1(NoL, k) * _G1(NoV, k);
}

double G_min(double NoH, double NoV, double NoL, double VoH){
    double g1 = 2.0 * NoH * NoV / VoH;
    double g2 = 2.0 * NoH * NoL / VoH;
    return std::min(1.0, std::min(g1, g2));
}

double Fresnel(double F0, double VoH){
    return F0 + (1.0 - F0) * pow(1 - VoH, 5.0);
    //double coeff = VoH * (-5.55473 * VoH - 6.98316);
    //return F0 + (1.0 - F0) * pow(2.0, coeff);

}

Eigen::Vector3d disney(Eigen::Vector3d l, Eigen::Vector3d v, 
                       Eigen::Vector3d n, Eigen::Vector3d d, 
                       double r, double s){
    
    Eigen::Vector3d h = (l + v) / 2.0;
    h.normalize();
    
    double NoH = maximum(n.dot(h), 0.0);
    double NoL = maximum(n.dot(l), 0.0);
    double NoV = maximum(n.dot(v), 0.0);
    double VoH = maximum(v.dot(h), 0.0);
    
    Eigen::Vector3d f_d = d / PI;
    
    double D = D_BlinnPhong(NoH, r);      //D_GGX(NoH, r);
    double G = G_min(NoH, NoV, NoL, VoH); //G_Smith(NoV, NoL, r);
    double F = Fresnel(0.1, VoH);  
    
    double tmp = D * G * F / (4.0 * NoL * NoV + 1e-12);
    
    Eigen::Vector3d f_s(tmp, tmp, tmp);
    
    return (f_d + 10 * s * f_s) * NoL * PI;
}


#endif
