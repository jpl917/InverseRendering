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

#include "common.h"

using namespace std;

double clamp(double c, double lower = 0.0, double upper = 1.0){
    if(c < lower) return lower;
    if(c > upper) return upper;
    return c;
}

double factorial(int n){  // 1*2*3*...*n
    int temp = 1;
    for(int i = 1; i <= n; i++){
        temp *= i;
    }
    return temp;
}

double K(int l, int m)
{
    // renormalisation constant for SH function
    double temp = ((2.0*l+1.0)*factorial(l-m)) / (4.0*PI*factorial(l+m));
    return sqrt(temp);
}

double P(int l,int m,double x)
{
    // evaluate an Associated Legendre Polynomial P(l,m,x) at x
    double pmm = 1.0;
    if(m>0) {
        double somx2 = sqrt((1.0-x)*(1.0+x));
    
        double fact = 1.0;
        for(int i=1; i<=m; i++) {
            pmm *= (-fact) * somx2;
            fact += 2.0;
        }
    }
    if(l==m) 
        return pmm;
    
    double pmmp1 = x * (2.0*m+1.0) * pmm;
    if(l==m+1) 
        return pmmp1;
    
    double pll = 0.0;
    for(int ll=m+2; ll<=l; ++ll) {
        pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

double SH(int m, int l, double phi, double theta)
{
    // return a point sample of a Spherical Harmonic basis function
    // l is the band, range [0..N]
    // m in the range [-l..l]
    // theta in the range [0..Pi]
    // phi in the range [0..2*Pi]
    const double sqrt2 = sqrt(2.0);
    if(m==0) 
        return K(l,0)*P(l,m,cos(theta));
    else if(m>0) 
        return sqrt2*K(l,m)*cos(m*phi)*P(l,m,cos(theta));
    else 
        return sqrt2*K(l,-m)*sin(-m*phi)*P(l,-m,cos(theta));
}


// phi: azimuth angle
// theta: polar angle
void sh_basis(double phi, double theta, std::vector<double>& coeff, int order=3){
    for(int l = 0; l < order; l++){
        for(int m = -l; m < l+1; m++){
            coeff.push_back(SH(m, l, phi, theta));
        }
    }
    return;
}


void cart2sph(double nx, double ny, double nz, double& phi, double& theta){
    if(nx==0){
        if(ny<0) phi = -PI/2;
        else phi = PI/2;
    }
            
    else{
        double temp = ny/nx;
        if(nx>0)  phi = atan(temp);
        else if(ny<0) phi = atan(temp) - PI;
        else phi = atan(temp) + PI;
            
    }
    //phi = np.arctan(y/x) 
    theta = acos(nz); //arccos(z)
    return;
}


void cal_SH_from_envmap(const std::string& filename, std::vector<std::vector<double> >& sh_coeff, const int& order = 3){
    
    cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED);
    img.convertTo(img, CV_32FC3, 1.0/255, 0);
    
    int order2 = order * order;
    
    int h = img.rows;
    int w = img.cols;
    
    // phi: azimuth angle
    // theta: polar angle
    vector<double> phi(w), theta(h);
    for(int i=0; i<w; i++) phi[i] = (i + 0.5) * 2 * PI / w;
    for(int i=0; i<h; i++) theta[i] = (i + 0.5) * PI / h;
    
    vector<double> w_phi(w), w_theta(h);
    for(int i=0; i<w; i++) w_phi[i] = (i+1)*2*PI/w - i*2*PI/w;
    for(int i=0; i<h; i++) w_theta[i] = cos(i*PI/h) - cos((i+1)*PI/h);
    
    
    std::vector<double> r_coeff(order2, 0.0), g_coeff(order2, 0.0), b_coeff(order2, 0.0);
    
    //int offset = w/4;
    
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            double d_omega_v = w_theta[i] * w_phi[j];
            
            std::vector<double> coeff;
            sh_basis(phi[j], theta[i], coeff, order);
            
            
            double b = img.at<cv::Vec3f>(i, j)[0];
            double g = img.at<cv::Vec3f>(i, j)[1];
            double r = img.at<cv::Vec3f>(i, j)[2];
            
            
            for(int k=0; k<order2; k++){
                b_coeff[k] += b * d_omega_v * coeff[k];
                g_coeff[k] += g * d_omega_v * coeff[k];
                r_coeff[k] += r * d_omega_v * coeff[k];
            }
            //cout<<d_omega_v<<endl;
            
        }
    }
    
    sh_coeff.resize(3);
    
    sh_coeff[0] = r_coeff;
    sh_coeff[1] = g_coeff;
    sh_coeff[2] = b_coeff;
    
    
    
#ifdef DEBUG
    cv::Mat approximate = cv::Mat::zeros(h, w, CV_8UC3);
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            std::vector<double> coeff;
            sh_basis(phi[j], theta[i], coeff, order);
            
            double r = 0, g = 0, b = 0;
            for(int k=0; k<order2; k++){
                r += coeff[k] * r_coeff[k];
                g += coeff[k] * g_coeff[k];
                b += coeff[k] * b_coeff[k];
            }
            
            approximate.at<cv::Vec3b>(i,j) = cv::Vec3b(255*clamp(b), 255*clamp(g), 255*clamp(r));
        }
    }
    cv::imwrite("approximate.jpg", approximate);
    
    
    int radius = 100;
    cv::Mat shading = cv::Mat::zeros(2*radius, 2*radius, CV_8UC3);
    for(int i = 0; i < 2 * radius; i++){
        double x = 1.0 * (i  - radius) / radius;
        
        for(int j = 0; j < 2 * radius; j++){
            double y = 1.0 * (radius - j) / radius;
            
            double dist = 1 - x * x - y * y;
            
            if(dist < 0) continue;
            
            std::vector<double> coeff;
            sh_basis(x, y, coeff, order);
            
            double r = 0, g = 0, b = 0;
            for(int k=0; k<order2; k++){
                r += coeff[k] * r_coeff[k];
                g += coeff[k] * g_coeff[k];
                b += coeff[k] * b_coeff[k];
            }
            
            shading.at<cv::Vec3b>(j, i) = cv::Vec3b(255*clamp(b), 255*clamp(g), 255*clamp(r));
        }
    }
    cv::imwrite("shading.jpg", shading);
#endif 
    
    
    
    return;
}


#endif
