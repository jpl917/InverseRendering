#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "headers.h"

class Visualizer{
public:
    Visualizer(){}
    ~Visualizer(){}
    
    
    // CV_32FC3 -> CV_8UC3 
    // [-1, 1]  -> [0, 1]
    static void visualize_normal_map(const cv::Mat& normal_map, const std::string& save_name);


    // visualize the position map 
    // normalize to [0,1] according to its bouding box
    static void visualize_position_map(const cv::Mat& position_map, 
                                const std::string& save_name);


    static void visualize_depth_map(const cv::Mat& depth_img, 
                            const std::string& save_name);


    static void visualize_render_image(const cv::Mat& render_img, 
                                const std::string& save_name);
    
};



#endif
