#ifndef RENDER_UTILS_H
#define RENDER_UTILS_H

#include "data_loader.h"
#include "spherical_harmonics.h"
#include "cook_torrance.h"
//#include "median_cut.h"

// point (x,y) 
class mypoint
{
 public:
    float x, y;

	mypoint(){}

	mypoint(float _x, float _y)
	{
		this->x = _x;
		this->y = _y;
	}

    float dot(mypoint p)
    {
        return this->x * p.x + this->y * p.y;
    }

    mypoint operator-(const mypoint& p)
    {
        mypoint np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        return np;
    }

    mypoint operator+(const mypoint& p)
    {
        mypoint np;
        np.x = this->x + p.x;
        np.y = this->y + p.y;
        return np;
    }

    mypoint operator*(float s)
    {
        mypoint np;
        np.x = s * this->x;
        np.y = s * this->y;
        return np;
    }
};


class RenderUtil
{

public:
    RenderUtil(){}
    ~RenderUtil(){}
    
    static bool isPointInTri(mypoint p, mypoint p0, mypoint p1, mypoint p2, std::vector<float>& weight);
    
    static void render_depth(const trimesh::TriMesh* mesh,
                    const Eigen::Matrix4d& projMatrix,
                    const int& img_h, const int& img_w,
                    cv::Mat& depth_img);



    static void convert_view_space_to_texture_space(trimesh::TriMesh* mesh,
                                                const vector<float>& vertexCoordinates,
                                                const vector<float>& textureCoordinates,
                                                const vector<int>& vertexIndices,
                                                const vector<int>& textureIndices,
                                                const Eigen::Matrix4d& projMatrix,
                                                const cv::Mat& depth_img,
                                                const cv::Mat& image_view_space,
                                                const int& texture_size,
                                                cv::Mat& image_texture_space, 
                                                cv::Mat& mask_texture_space);


    // preprocess the normal map and position map
    static void render_normal_position(const trimesh::TriMesh* mesh,
                                const vector<float>& textureCoordinates,
                                const vector<int>& vertexIndices,
                                const vector<int>& textureIndices,
                                const int& h, const int& w, const int& c,
                                cv::Mat& normal_map, cv::Mat& position_map, cv::Mat& triangle_idx_map);
                            

    
    // render a view using the 4 reflectance maps
    static void render_image_view_space(const trimesh::TriMesh* mesh,
                    const vector<float>& textureCoordinates,
                    const vector<int>& textureIndices,
                    const Eigen::Matrix4d& projMatrix,
                    const std::vector<Eigen::Matrix3d>& face_TBN,
                    const std::vector<Eigen::Matrix3d>& face_TBN_norm,
                    const cv::Mat& diff_albedo,
                    const cv::Mat& spec_albedo,
                    const cv::Mat& roughness_map,
                    const cv::Mat& displacement_map,
                    const std::vector<float3>& lights_pos,
                    const std::vector<float3>& lights_color,
                    const int& img_h, const int& img_w, const int& img_c,
                    cv::Mat& image);
    
    
    // render a view using image stored in texture space
    static void render_image_view_space(const trimesh::TriMesh* mesh,
                    const vector<float>& textureCoordinates,
                    const vector<int>& textureIndices,
                    const Eigen::Matrix4d& projMatrix,
                    const cv::Mat& image_texture_space,
                    const int& img_h, const int& img_w, const int& img_c,
                    cv::Mat& image);


    // render a view in texture space
    static void render_image_texture_space(const std::vector<Eigen::Matrix3d>& face_TBN,
                                    const std::vector<Eigen::Matrix3d>& face_TBN_norm,
                                    const cv::Mat& normal_map,
                                    const cv::Mat& triangle_idx_map,
                                    const cv::Mat& mask_texture_space,
                                    const cv::Mat& diff_albedo,
                                    const cv::Mat& spec_albedo,
                                    const cv::Mat& roughness_map,
                                    const cv::Mat& displacement_map,
                                    const std::vector<float3>& lights_pos,
                                    const std::vector<float3>& lights_color,
                                    const int& texture_size,
                                    cv::Mat& image_ts);
    
};


#endif
