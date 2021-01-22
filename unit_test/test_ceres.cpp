#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/loss_function.h"

#include "pgm_image.h"

using namespace std;

using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;


struct RenderResidual{
    RenderResidual(const double& render_value):m_render_value(render_value){}
    
    template <typename T>
    bool operator()(const T* const alpha, T* residual)const{
        T rho_d = T(alpha[0]);
        
        residual[0] = pow(m_render_value - rho_d, 2);
        
        return true;
    }
    
private:
    const double m_render_value;
};


struct SmoothConstraint{
    SmoothConstraint(){}
    
    template<typename T>
    bool operator()(const T* const center, 
                    const T* const up, 
                    const T* const down, 
                    const T* const left, 
                    const T* const right, 
                    T* residual)const{
        residual[0] = 0.1 * pow(T(5.0) * T(center[0]) - T(up[0]) - T(down[0]) - T(left[0]) - T(right[0]), 2);
        return true;
    }
};

int main()
{
//     cv::Mat img = cv::imread("../a.png");
//     int img_w  = img.cols;
//     int img_h = img.rows;
//     int img_c = img.channels();
//     
//     cout<<img_c<<endl;
    
    
    ceres::examples::PGMImage<double> image("../imgs/1.pgm");
    
    image *= 1/255.0;
    
    ceres::examples::PGMImage<double> solution(image.width(), image.height());
    solution.Set(0);
    
    ceres::Problem problem;
    for(int x = 0; x < 100; x++){      //image.width()
        for(int y = 0; y < 100; y++){  //image.height()
            double value = image.Pixel(x, y);
            ceres::CostFunction* renderCostFunc = new AutoDiffCostFunction<RenderResidual, 1, 1>(new RenderResidual(value));
            
            problem.AddResidualBlock(renderCostFunc, NULL, (&solution)->MutablePixelFromLinearIndex(y * image.width() + x));
            
            
            if( x > 0 && x < 100 && y > 0 && y < 100){
                ceres::CostFunction* smoothCostFunc = new AutoDiffCostFunction<SmoothConstraint, 1, 1,1,1,1,1>(new SmoothConstraint());
                
                
//                 double* parameters_pointers[] = {(&solution)->MutablePixelFromLinearIndex(y     * image.width() + x),
//                                                  (&solution)->MutablePixelFromLinearIndex((y-1) * image.width() + x),
//                                                  (&solution)->MutablePixelFromLinearIndex((y+1) * image.width() + x),
//                                                  (&solution)->MutablePixelFromLinearIndex( y    * image.width() + x - 1),
//                                                  (&solution)->MutablePixelFromLinearIndex( y    * image.width() + x + 1)};
//                 
//                 problem.AddResidualBlock(costFunc, NULL, parameters_pointers);
              
                problem.AddResidualBlock(smoothCostFunc, NULL, 
                                                 (&solution)->MutablePixelFromLinearIndex(y     * image.width() + x),
                                                 (&solution)->MutablePixelFromLinearIndex((y-1) * image.width() + x),
                                                 (&solution)->MutablePixelFromLinearIndex((y+1) * image.width() + x),
                                                 (&solution)->MutablePixelFromLinearIndex( y    * image.width() + x - 1),
                                                 (&solution)->MutablePixelFromLinearIndex( y    * image.width() + x + 1));
            }
            
        }
    }
    
     
    
    ceres::Solver::Options options;
    //options.max_num_iterations = 10;
    options.linear_solver_type = ceres::DENSE_QR; 
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 1;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    
    solution *= 255.0;
    //solution.WriteToFile("../aa.pgm");
    
    
    cv::Mat ret(image.height(), image.width(), CV_8UC1);
    for(int x=0; x<image.height(); x++){
        for(int y=0; y<image.width(); y++){
            ret.at<uchar>(x,y) = int(solution.Pixel(y, x));
        }
    }
    cv::imwrite("../imgs/1-a.png", ret);
    
    
    return 0;
    
//     int num_params = 3 * img_width * img_height;
//     cout<<num_params<<endl;
//     
//     double* w = (double*)malloc(num_params);
//     for(int i=0; i<num_params; i++) w[i] = 0;
//     cout<<"aaa"<<endl;
//     
//     
//     
//     
//     
//     cout << summary.BriefReport() << endl;
//     
//     cv::Mat ret(img.rows, img.cols, CV_8UC3);
//     for(int i=0; i<img.rows; i++){
//         for(int j=0; j<img.cols; j++){
//             int idx = 3 * (i * img.cols + j);
//             ret.at<cv::Vec3b>(i,j) = cv::Vec3b(w[idx], w[idx+1], w[idx+2]);
//         }
//     }
//     cv::imwrite("ret.jpg",ret);
//     
//     return 0;
   
    
    
//     ceres::Problem problem;
//     ceres::CostFunction* costFunc = new AutoDiffCostFunction<RenderResidual, 3, 2>(new RenderResidual(10));
//     
//     double w[2] = {0, 0};
//     problem.AddResidualBlock(costFunc, NULL, w);
//     
//     ceres::Solver::Options options;
//     options.max_num_iterations = 100;
//     options.linear_solver_type = ceres::DENSE_QR;
//     options.minimizer_progress_to_stdout = false;
//     options.num_threads = 8;
//     
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);
//     
//     cout << summary.BriefReport() << endl;
//     cout<<w[0]<<"  "<<w[1]<<endl;    
//     return 0;
}

