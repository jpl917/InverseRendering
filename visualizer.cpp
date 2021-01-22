#include "visualizer.h"

// CV_32FC3 -> CV_8UC3 
// [-1, 1]  -> [0, 1]
void Visualizer::visualize_normal_map(const cv::Mat& normal_map, const std::string& save_name){
    int h = normal_map.rows;  
    int w = normal_map.cols;
    cv::Mat viz = cv::Mat::zeros(h, w, CV_8UC3);
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            cv::Vec3f val = normal_map.at<cv::Vec3f>(i,j);
            if( val[0] == 0 && val[1] == 0 && val[2] == 0) continue;
            
            viz.at<cv::Vec3b>(i,j) = cv::Vec3b(
                                    (val[0] + 1.0) / 2.0 * 255, 
                                    (val[1] + 1.0) / 2.0 * 255, 
                                    (val[2] + 1.0) / 2.0 * 255);
        }
    }
    cv::imwrite(save_name, viz);
}

// visualize the position map 
// normalize to [0,1] according to its bouding box
void Visualizer::visualize_position_map(const cv::Mat& position_map, 
                            const std::string& save_name)
{    
    int h = position_map.rows;  
    int w = position_map.cols;
    
    // calculate the bounding box
    vector<double> min_xyz(3, 1e10), max_xyz(3, -1e10);
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            cv::Vec3f v = position_map.at<cv::Vec3f>(i,j);
            if((v[0] + v[1] + v[2]) == 0)continue;
            
            for(int k=0; k<3; k++){
                min_xyz[k] = min_xyz[k] < v[k] ? min_xyz[k] : v[k];
                max_xyz[k] = max_xyz[k] > v[k] ? max_xyz[k] : v[k];
            }
            //fout<<"v "<<v[0]<<" "<<v[1]<<" "<<v[2]<<endl;
        }
    }
    cout<<"Bounding box "<<endl;
    cout<<min_xyz[0]<<" "<<max_xyz[0]<<endl;
    cout<<min_xyz[1]<<" "<<max_xyz[1]<<endl;
    cout<<min_xyz[2]<<" "<<max_xyz[2]<<endl;
    
    
    // save to 8UC3
    cv::Mat viz = cv::Mat::zeros(h, w, CV_8UC3);
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            cv::Vec3f v = position_map.at<cv::Vec3f>(i,j);
            if((v[0] + v[1] + v[2]) == 0)continue;
            
            viz.at<cv::Vec3b>(i,j) = cv::Vec3b(
                                    (v[0] - min_xyz[0]) / (max_xyz[0] - min_xyz[0]) * 255, 
                                    (v[1] - min_xyz[1]) / (max_xyz[1] - min_xyz[1]) * 255, 
                                    (v[2] - min_xyz[2]) / (max_xyz[2] - min_xyz[2]) * 255);
        }
    }
    cv::imwrite(save_name, viz);
}



void Visualizer::visualize_depth_map(const cv::Mat& depth_img, 
                         const std::string& save_name)
{
    int img_h = depth_img.rows;
    int img_w = depth_img.cols;
    
    double max_val = -1e10, min_val = 1e10;
    for(int i=0; i<img_h; i++)
    {
        for(int j=0; j<img_w; j++)
        {
            if(depth_img.at<float>(i,j) > max_val) max_val = depth_img.at<float>(i,j);
            if(depth_img.at<float>(i,j) < min_val && depth_img.at<float>(i,j) != -99999)
            {
                min_val = depth_img.at<float>(i,j);
            }
        }
    }
    
    
    cv::Mat viz = cv::Mat::zeros(img_h, img_w, CV_8UC1);
    for(int i=0; i<img_h; i++){
        for(int j=0; j<img_w; j++){
            if(depth_img.at<float>(i,j) != -99999) 
                viz.at<uchar>(i,j)= 255 * (depth_img.at<float>(i,j) - min_val)/(max_val-min_val);
        }
    }
    
    cv::imwrite(save_name, viz);
}


void Visualizer::visualize_render_image(const cv::Mat& render_img, 
                            const std::string& save_name)
{
    int img_h = render_img.rows;
    int img_w = render_img.cols;
    
    
    cv::Mat viz = cv::Mat::zeros(img_h, img_w, CV_8UC3);
    for(int i=0; i<img_h; i++){
        for(int j=0; j<img_w; j++){
            cv::Vec3b val(0,0,0);
            for(int k=0; k<3; k++){
                val[k] = 255 * pow(render_img.at<cv::Vec3f>(i,j)[k], 1/2.2);
            }
            viz.at<cv::Vec3b>(i,j) = val;
        }
    }
    
    cv::imwrite(save_name, viz);
}
