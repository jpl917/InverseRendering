#ifndef RENDER_H
#define RENDER_H

#include "utils.h"
#include "spherical_harmonics.h"
#include "cook_torrance.h"

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



bool isPointInTri(mypoint p, mypoint p0, mypoint p1, mypoint p2, std::vector<float>& weight)
{
    // vectors
    mypoint v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    // dot products
    float dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    float dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    float dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    float dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    float dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    float inverDeno;
    if(dot00*dot11 - dot01*dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1/(dot00*dot11 - dot01*dot01);

    float u = (dot11*dot02 - dot01*dot12)*inverDeno;
    float v = (dot00*dot12 - dot01*dot02)*inverDeno;

    //weight
    weight.resize(3);
    weight[0] = 1 - u - v;
    weight[1] = v;
    weight[2] = u;

    // check if point in triangle
    return (u >= 0) && (v >= 0) && (u + v < 1);
}


void renderTexture(//const trimesh::TriMesh* mesh,
                   const vector<float>& colors,
                   const vector<float>& textureCoordinates,
                   const vector<int>& triangles,
                   const vector<int>& triangles_uv,
                   std::vector<float>& image,
                   const int& h, const int& w, const int& c){
    std::vector<float> depth_buffer(h*w, 0);
    
    size_t ntri = triangles_uv.size()/3;
    
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    mypoint p, p0, p1, p2;
    vector<float> weight;
    float x_min, x_max, y_min, y_max;
    float p_color, p0_color, p1_color, p2_color;
    
    for(size_t i = 0; i < ntri; i++)
    {
        tri_p0_ind = triangles_uv[3*i];
        tri_p1_ind = triangles_uv[3*i+1];
        tri_p2_ind = triangles_uv[3*i+2];


        p0.x = textureCoordinates[3*tri_p0_ind]; p0.y = textureCoordinates[3*tri_p0_ind + 1]; 
        p1.x = textureCoordinates[3*tri_p1_ind]; p1.y = textureCoordinates[3*tri_p1_ind + 1]; 
        p2.x = textureCoordinates[3*tri_p2_ind]; p2.y = textureCoordinates[3*tri_p2_ind + 1]; 

		
		tri_p0_ind = triangles[3*i];
		tri_p1_ind = triangles[3*i+1];
        tri_p2_ind = triangles[3*i+2];

        // bounding box of the triangle
        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if(x_min > x_max || y_min > y_max) continue;

        for(int y = y_min; y <= y_max; y++)
        {
            for(int x = x_min; x <= x_max; x++)
            {
                p.x = x;
                p.y = y;

                if(isPointInTri(p, p0, p1, p2, weight))
                {
                   
                    if(depth_buffer[y * w + x] != 1)
                    {
                        for(int k = 0; k < c; k++) //for each color channel  RGB
                        {
                            p0_color = colors[c*tri_p0_ind + k];
                            p1_color = colors[c*tri_p1_ind + k];
                            p2_color = colors[c*tri_p2_ind + k];
					
                            p_color = weight[0]*p0_color + weight[1]*p1_color + weight[2]*p2_color;

                            image[c*(y*w+x) + k] = p_color;
                        }

                        depth_buffer[y * w + x] = 1;
                        //cout<<depth_buffer[y * w + c]<<endl;
                    }


                }
            }
        }
        //process each triangle end

    }
}


// calculate the new normal map from the displacement map
void renderNormalMap(const trimesh::TriMesh* mesh,
                   const vector<float>& textureCoordinates,
                   const vector<int>& vertexIndices,
                   const vector<int>& textureIndices,
                   const std::vector<Eigen::Matrix3d>& face_TBN,
                   const std::vector<Eigen::Matrix3d>& face_TBN_norm,
                   const cv::Mat& displacement_map,
                   cv::Mat& normal_map_detailed,
                   const int& h, const int& w, const int& c)
{
    normal_map_detailed = cv::Mat::zeros(h, w, CV_32FC3);
    
    std::vector<float> depth_buffer(h*w, 0);
    
    size_t ntri = textureIndices.size()/3;
    
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    mypoint p, p0, p1, p2;
    vector<float> weight;
    float x_min, x_max, y_min, y_max;
    float p_color, p0_color, p1_color, p2_color;
    
    // loop each triangle
    for(size_t i = 0; i < ntri; i++)
    {
        tri_p0_ind = textureIndices[3*i];
        tri_p1_ind = textureIndices[3*i+1];
        tri_p2_ind = textureIndices[3*i+2];


        p0.x = textureCoordinates[3*tri_p0_ind]; p0.y = textureCoordinates[3*tri_p0_ind + 1]; 
        p1.x = textureCoordinates[3*tri_p1_ind]; p1.y = textureCoordinates[3*tri_p1_ind + 1]; 
        p2.x = textureCoordinates[3*tri_p2_ind]; p2.y = textureCoordinates[3*tri_p2_ind + 1]; 

		
		tri_p0_ind = vertexIndices[3*i];
		tri_p1_ind = vertexIndices[3*i+1];
        tri_p2_ind = vertexIndices[3*i+2];
        

        // bounding box of the triangle
        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if(x_min > x_max || y_min > y_max) continue;

        for(int y = y_min; y <= y_max; y++)
        {
            for(int x = x_min; x <= x_max; x++)
            {
                p.x = x;
                p.y = y;

                if(isPointInTri(p, p0, p1, p2, weight))
                {
                   
                    if(depth_buffer[y * w + x] != 1)
                    {
                        Eigen::Vector3d interpolate_normal;
                        for(int k = 0; k < c; k++) //for each color channel  RGB
                        {
                            p0_color = mesh->normals[tri_p0_ind][k]; 
                            p1_color = mesh->normals[tri_p1_ind][k]; 
                            p2_color = mesh->normals[tri_p2_ind][k]; 
					
                            interpolate_normal[k] = weight[0]*p0_color + weight[1]*p1_color + weight[2]*p2_color;
                        }
                        
                        interpolate_normal.normalize();
                        
                        
                        // displacement map
                        double du = displacement_map.at<cv::Vec3f>(y, x)[0];
                        double dv = displacement_map.at<cv::Vec3f>(y, x)[0];
                        if(x >= 1) du = displacement_map.at<cv::Vec3f>(y, x)[0] - displacement_map.at<cv::Vec3f>(y, x-1)[0];
                        if(y >= 1) dv = displacement_map.at<cv::Vec3f>(y, x)[0] - displacement_map.at<cv::Vec3f>(y-1, x)[0];
                            
                        Eigen::Vector3d duv(-4096*du, -4096*dv, 1.0);
                        
                        Eigen::Matrix3d tbn;
                        tbn << face_TBN[i](0,0), face_TBN[i](0,1), interpolate_normal[0],
                               face_TBN[i](1,0), face_TBN[i](1,1), interpolate_normal[1],
                               face_TBN[i](2,0), face_TBN[i](2,1), interpolate_normal[2];
                        
                        Eigen::Vector3d n = tbn * face_TBN_norm[i] * duv;
                        n.normalize();
                        
                        for(int k=0; k<c; k++){
                            //image[c*(y*w+x) + k] = n[k]; //interpolate_normal[k]
                            
                            normal_map_detailed.at<cv::Vec3f>(y,x)[k] = n[k]; //(n[k] + 1.0) / 2.0 * 255;  
                        }


                        depth_buffer[y * w + x] = 1;
                    }

                }
            }
        }
        //process each triangle 

    }
    
    //cv::imwrite("debug.jpg", normalMap);
}




void renderImage(const trimesh::TriMesh* mesh,
                 const vector<float>& vertices,
                 const vector<float>& textureCoordinates,
                 const vector<int>& vertexIndices,
                 const vector<int>& textureIndices,
                 const std::vector<Eigen::Matrix3d>& face_TBN,
                 const std::vector<Eigen::Matrix3d>& face_TBN_norm,
                 const cv::Mat& diff_albedo,
                 const cv::Mat& spec_albedo,
                 const cv::Mat& roughness_map,
                 const cv::Mat& displacement_map,
                 cv::Mat& normal_map_detailed,
                 const std::vector<std::vector<double> >& sh_coeffs,
                 std::vector<float>& image_buffer,
                 const int& h, const int& w, const int& c)
{
    std::vector<float> depth_buffer(h * w, -1e10);
    
    
    size_t ntri = vertexIndices.size()/3;

    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    int tex_p0_ind, tex_p1_ind, tex_p2_ind;
    mypoint p, p0, p1, p2;
    vector<float> weight;
    float x_min, x_max, y_min, y_max;
    float p_depth, p0_depth, p1_depth, p2_depth;
    
    mypoint p_uv, p0_uv, p1_uv, p2_uv;

    for(size_t i = 0; i < ntri; i++)
    {

        tri_p0_ind = vertexIndices[3*i];
        tri_p1_ind = vertexIndices[3*i+1];
        tri_p2_ind = vertexIndices[3*i+2];

        // 2D projection
        p0.x = vertices[3*tri_p0_ind]; p0.y = vertices[3*tri_p0_ind + 1]; p0_depth = vertices[3*tri_p0_ind + 2];
        p1.x = vertices[3*tri_p1_ind]; p1.y = vertices[3*tri_p1_ind + 1]; p1_depth = vertices[3*tri_p1_ind + 2];
        p2.x = vertices[3*tri_p2_ind]; p2.y = vertices[3*tri_p2_ind + 1]; p2_depth = vertices[3*tri_p2_ind + 2];
        
        
        // uv coordinates
        tex_p0_ind = textureIndices[3*i];
        tex_p1_ind = textureIndices[3*i+1];
        tex_p2_ind = textureIndices[3*i+2];
        
//         cv::Vec3f color0 = diff_albedo.at<cv::Vec3f>(textureCoordinates[3*tri_p0_ind+1], textureCoordinates[3*tri_p0_ind]);
//         cv::Vec3f color1 = diff_albedo.at<cv::Vec3f>(textureCoordinates[3*tri_p1_ind+1], textureCoordinates[3*tri_p1_ind]);
//         cv::Vec3f color2 = diff_albedo.at<cv::Vec3f>(textureCoordinates[3*tri_p2_ind+1], textureCoordinates[3*tri_p2_ind]);
        p0_uv.x = textureCoordinates[3*tex_p0_ind]; p0_uv.y = textureCoordinates[3*tex_p0_ind+1];
        p1_uv.x = textureCoordinates[3*tex_p1_ind]; p1_uv.y = textureCoordinates[3*tex_p1_ind+1];
        p2_uv.x = textureCoordinates[3*tex_p2_ind]; p2_uv.y = textureCoordinates[3*tex_p2_ind+1];
        

        // bounding box of the triangle
        x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
        x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);

        y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
        y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if(x_min > x_max || y_min > y_max) continue;

        for(int y = y_min; y <= y_max; y++)
        {
            for(int x = x_min; x <= x_max; x++)
            {
                p.x = x; p.y = y;

                if(isPointInTri(p, p0, p1, p2, weight))
                {
                    p_depth = weight[0] * p0_depth + weight[1] * p1_depth + weight[2] * p2_depth;

                    if(fabs(p_depth) < fabs(depth_buffer[y * w + x]))
                    {
                        depth_buffer[y * w + x] = p_depth;
                        
                        // the corresponding texture coordinate
                        p_uv.x = weight[0] * p0_uv.x + weight[1] * p1_uv.x + weight[2] * p2_uv.x;
                        p_uv.y = weight[0] * p0_uv.y + weight[1] * p1_uv.y + weight[2] * p2_uv.y;
                        
                        cv::Vec3f diff_color = diff_albedo.at<cv::Vec3f>(p_uv.y, p_uv.x);
                        Eigen::Vector3d d(diff_color[0], diff_color[1], diff_color[2]);
                        double r = roughness_map.at<cv::Vec3f>(p_uv.y, p_uv.x)[0];
                        double s = spec_albedo.at<cv::Vec3f>(p_uv.y, p_uv.x)[0];
                        
                        
                        // interpolate normal for smoothness
                        Eigen::Vector3d interpolate_normal;
                        for(int k = 0; k < c; k++) //for each color channel RGB
                        {
                            interpolate_normal[k] = weight[0]*mesh->normals[tri_p0_ind][k] + 
                                                    weight[1]*mesh->normals[tri_p1_ind][k] + 
                                                    weight[2]*mesh->normals[tri_p2_ind][k];
                        }
                        interpolate_normal.normalize();
                        
                        double du = displacement_map.at<cv::Vec3f>(p_uv.y, p_uv.x)[0];
                        double dv = displacement_map.at<cv::Vec3f>(p_uv.y, p_uv.x)[0];
                        if(p_uv.x >= 1) du = displacement_map.at<cv::Vec3f>(p_uv.y, p_uv.x)[0] - 
                                            displacement_map.at<cv::Vec3f>(p_uv.y, p_uv.x-1)[0];
                        if(p_uv.y >= 1) dv = displacement_map.at<cv::Vec3f>(p_uv.y, p_uv.x)[0] -     
                                            displacement_map.at<cv::Vec3f>(p_uv.y-1, p_uv.x)[0];
                            
                        Eigen::Vector3d duv(-4096*du, -4096*dv, 1.0);
                        
                        Eigen::Matrix3d tbn;
                        tbn << face_TBN[i](0,0), face_TBN[i](0,1), interpolate_normal[0],
                               face_TBN[i](1,0), face_TBN[i](1,1), interpolate_normal[1],
                               face_TBN[i](2,0), face_TBN[i](2,1), interpolate_normal[2];
                        
                        Eigen::Vector3d n = tbn * face_TBN_norm[i] * duv; 
                        //n = interpolate_normal;
                        n.normalize();
                        
                        normal_map_detailed.at<cv::Vec3f>(p_uv.y, p_uv.x) = cv::Vec3f(n[0], n[1], n[2]);
                        
                        // useless
                        double phi, theta;
                        cart2sph(n[0], n[1], n[2], phi, theta);
                        std::vector<double> coeff;
                        sh_basis(phi, theta, coeff, SH_ORDER);
                        
                        

                        Eigen::Vector3d ret(0,0,0);
                        {
                        
                            Eigen::Vector3d l(10,10,10);
                            Eigen::Vector3d v(0,0,100);
                            l.normalize();
                            v.normalize();
                            ret = disney(l, v, n, d, r, s);
                        }
                        
                        {
                            Eigen::Vector3d l(-1,0,0);
                            Eigen::Vector3d v(0,0,100);
                            l.normalize();
                            v.normalize();
                            ret += 0.2 * disney(l, v, n, d, r, s);
                        }
                        
                        for(int k=0; k<3; k++){
                            if(ret[k] > 1.0) ret[k] = 1.0;
                            if(ret[k] < 0.0) ret[k] = 0.0;
                        }
                        
                        
                        // spherical hamonics lighting
                        for(int k=0; k<3; k++){
//                             double light_tmp = 0.0;
//                             for(int q = 0; q < SH_ORDER * SH_ORDER; q++){
//                                 light_tmp += coeff[q] * sh_coeffs[2-k][q];  // bgr
//                             }
//                             image_buffer[3 * (y * w + x) + k] = diff_color[k] * light_tmp;
                            
                            
                            image_buffer[3 * (y * w + x) + k] = ret[k];
                        }
                    }
                }
            }
        }
    }
}


#endif
