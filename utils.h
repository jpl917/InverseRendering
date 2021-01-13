#ifndef UTILS_H
#define UTILS_H

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

void load_albedo()
{
    cv::Mat diffuse_albedo = cv::imread("../maya_render/Textures/Head_Roughness.exr", cv::IMREAD_UNCHANGED);
    
    int texSize = diffuse_albedo.cols; 
    assert(diffuse_albedo.cols == diffuse_albedo.rows);
    
    cout<<texSize<<" "<<diffuse_albedo.rows<<endl;
    
    cv::Mat ret = cv::Mat(texSize, texSize, CV_8UC3);
    for(int row = 0; row < texSize; row++){
        for(int col = 0; col < texSize; col++){
            cv::Vec3f val = diffuse_albedo.at<cv::Vec3f>(row,col);
            
            
            cv::Vec3f val1;
            val1[0] = pow(val[0], 1/2.2);
            val1[1] = pow(val[1], 1/2.2);
            val1[2] = pow(val[2], 1/2.2);
            ret.at<cv::Vec3b>(row, col) = 255*val1;//cv::Vec3b()
        }
    }
    
    cv::imwrite("ret.jpg", ret);
    return;
}

void loadMesh(const string& filename,
            std::vector<float>& vertexCoordinates,
            std::vector<float>& textureCoordinates,
            std::vector<int>&   vertexIndices,
            std::vector<int>&   textureIndices)
{
    
    int TEXTURE_SIZE = 4096;
    
    std::cout<<filename.c_str()<<std::endl;
    vertexCoordinates.clear();
    textureCoordinates.clear();
    vertexIndices.clear();
    textureIndices.clear();

    FILE * file = fopen(filename.c_str(), "r");
    if( file == NULL )
    {
        printf("Impossible to open the file !\n");
        return;
    }
    while(1){
        char lineHeader[128];
        int res = fscanf(file, "%s", lineHeader);
        if(res == EOF) break;

        if( strcmp( lineHeader, "v" ) == 0 ){
            float x=0, y=0, z=0;
            int matches = fscanf(file, "%f %f %f\n", &x, &y, &z);
            if (matches != 3){
                printf("v can't be read by our simple parser \n");
                return;
            }
             
            vertexCoordinates.push_back(x);
            vertexCoordinates.push_back(y);
            vertexCoordinates.push_back(z);
            
        }
        else if (strcmp( lineHeader, "vt" ) == 0 ){
            float u=0, v=0;
            int matches = fscanf(file, "%f %f\n", &u, &v);
            if (matches != 2){
                printf("vt can't be read by our simple parser \n");
                return;
            }
            
            u = u * (TEXTURE_SIZE-1);
			v = v * (TEXTURE_SIZE-1);
			v = TEXTURE_SIZE - 1 - v;

            textureCoordinates.push_back(u);
            textureCoordinates.push_back(v);
            textureCoordinates.push_back(0);

        }
        else if ( strcmp( lineHeader, "f" ) == 0 ){
            unsigned int vertexIndex[3], textureIndex[3]; //, normalIndex[3];
            int matches = fscanf(file, "%d/%d %d/%d %d/%d\n", &vertexIndex[0], &textureIndex[0], \
                                                              &vertexIndex[1], &textureIndex[1], \
                                                              &vertexIndex[2], &textureIndex[2] );
            if (matches != 6){
                printf("f can't be read by our simple parser \n");
                return;
            }

            vertexIndices.push_back(vertexIndex[0]-1);
            vertexIndices.push_back(vertexIndex[1]-1);
            vertexIndices.push_back(vertexIndex[2]-1);
			
            textureIndices.push_back(textureIndex[0]-1);
            textureIndices.push_back(textureIndex[1]-1);
            textureIndices.push_back(textureIndex[2]-1);
            
        }
    }
    
    fclose(file);
    
    return;
}


void loadCamera(const string& filename,
                std::vector<Eigen::Matrix4d>& cam_K,
                std::vector<Eigen::Matrix4d>& cam_T,
                std::vector<Eigen::Matrix4d>& cam_P){
    
    ifstream fin(filename);
    
    int camNum = 12;
    
    cam_K.resize(camNum);
    cam_T.resize(camNum);
    cam_P.resize(camNum);
    
    for(int i=0; i < camNum; i++){
        string tmp1, tmp2;
        
        fin>>tmp1>>tmp2;
        for(int j=0; j<16; j++) fin >> cam_K[i](j/4, j%4);
        
        fin>>tmp1>>tmp2;
        for(int j=0; j<16; j++) fin >> cam_T[i](j/4, j%4);
        
        fin>>tmp1>>tmp2;
        for(int j=0; j<16; j++) fin >> cam_P[i](j/4, j%4);
        
    }
    return;
}


// outside the image space -> false
bool proj_3D_to_2D(const trimesh::point& p, 
				   const Eigen::Matrix<double,3,4>& projMatrix,
				   float& x, float& y, float& z,
                   int img_height, int img_width) 
{
	Eigen::Vector3d pt(p.x,p.y,p.z);
	Eigen::Vector4d pt_h(p.x,p.y,p.z,1);
	
	Eigen::Vector3d uv_h = projMatrix * pt_h;	
	x = uv_h(0)/uv_h(2);  //(int)round()
	y = uv_h(1)/uv_h(2);
	z = uv_h(2);
	
	if(x < 0 || x >= img_width || y < 0 || y >= img_height)  return false;

	return true;
}



void calculateTBN(trimesh::TriMesh* mesh,
                  const std::vector<float>& uvCoordinates,
                  const std::vector<int>& uvIndices,
                  std::vector<Eigen::Matrix3d>& vertex_TBN,
                  std::vector<Eigen::Matrix3d>& face_TBN,
                  std::vector<Eigen::Matrix3d>& face_TBN_norm)
{

    int TEXTURE_SIZE = 4096;
	cout<<"Calculting TBN ..."<<endl;
	
    std::vector<trimesh::point> verts = mesh->vertices;
    std::vector<trimesh::TriMesh::Face> faces= mesh->faces;
    
    cout<<"vertex num: "<<verts.size() << "    face num: "<< faces.size()<<endl;

	face_TBN.resize(faces.size());
    
    face_TBN_norm.resize(faces.size());

    //calculate the TBN
    std::vector<Eigen::Vector3d> face_T(faces.size());
    std::vector<Eigen::Vector3d> face_B(faces.size());

    for(size_t i=0; i<faces.size(); i++)
    {
        //cout<<i<<endl;
        int idx0 = faces[i][0];
        int idx1 = faces[i][1];
        int idx2 = faces[i][2];

        trimesh::point vert0 = verts[idx0];
        trimesh::point vert1 = verts[idx1];
        trimesh::point vert2 = verts[idx2];

        float v1x = vert1[0] - vert0[0];
        float v1y = vert1[1] - vert0[1];
        float v1z = vert1[2] - vert0[2];

        float v2x = vert2[0] - vert0[0];
        float v2y = vert2[1] - vert0[1];
        float v2z = vert2[2] - vert0[2];
        
//         float v1v0Norm = sqrt(v1x*v1x+v1y*v1y+v1z*v1z);
//         v1x /= v1v0Norm;
//         v1y /= v1v0Norm;
//         v1z /= v1v0Norm;
//         
//         float v2v0Norm = sqrt(v2x*v2x+v2y*v2y+v2z*v2z);
//         v2x /= v2v0Norm;
//         v2y /= v2v0Norm;
//         v2z /= v2v0Norm;
        
        idx0 = uvIndices[3*i];
        idx1 = uvIndices[3*i+1];
        idx2 = uvIndices[3*i+2];

        float u1x = (uvCoordinates[3*idx1]   - uvCoordinates[3*idx0])   / float(TEXTURE_SIZE-1);
        float u1y = (uvCoordinates[3*idx1+1] - uvCoordinates[3*idx0+1]) / float(TEXTURE_SIZE-1);

        float u2x = (uvCoordinates[3*idx2]   - uvCoordinates[3*idx0])   / float(TEXTURE_SIZE-1);
        float u2y = (uvCoordinates[3*idx2+1] - uvCoordinates[3*idx0+1]) / float(TEXTURE_SIZE-1);

        float det = u1x * u2y - u2x * u1y;
        //for T
        float tx = (v1x * u2y - v2x * u1y) / det;
        float ty = (v1y * u2y - v2y * u1y) / det;
        float tz = (v1z * u2y - v2z * u1y) / det;

        //for B
        float bx = (-v1x * u2x + v2x * u1x) / det;
        float by = (-v1y * u2x + v2y * u1x) / det;
        float bz = (-v1z * u2x + v2z * u1x) / det;

        //for N
        float nx = by * tz - bz * ty;
        float ny = bz * tx - bx * tz;
        float nz = bx * ty - by * tx;

        face_T[i] = Eigen::Vector3d(tx, ty, tz);
        face_B[i] = Eigen::Vector3d(bx, by, bz);
		Eigen::Vector3d face_N = Eigen::Vector3d(nx,ny,nz); //face_T[i].cross(face_B[i]); //
		
		//normalized
		tx /= face_T[i].norm(); ty /= face_T[i].norm(); tz /= face_T[i].norm();
        bx /= face_B[i].norm(); by /= face_B[i].norm(); bz /= face_B[i].norm();
        face_N.normalize();
        
		
		face_TBN[i] << tx, bx, face_N[0],
					   ty, by, face_N[1],
					   tz, bz, face_N[2];
                       
        
        Eigen::Matrix3d tmp;
        tmp << face_T[i].norm(), 0, 0, 
               0, face_B[i].norm(), 0,
               0, 0, face_T[i].norm()*face_B[i].norm();

        face_TBN_norm[i] = tmp;
        
        if(i == 2000){
            cout<<face_T[i].norm()<<" "<<face_B[i].norm()<<endl;
            cout<<face_TBN_norm[i]<<endl<<endl;
        }
    }


    std::vector<Eigen::Vector3d> vertex_T(verts.size());
    std::vector<Eigen::Vector3d> vertex_B(verts.size());
    std::vector<Eigen::Vector3d> vertex_N(verts.size());

    vertex_TBN.resize(verts.size());
    mesh->need_adjacentfaces();
    std::vector<std::vector<int> > adjacentfaces = mesh->adjacentfaces;
    for(size_t i=0; i<verts.size(); i++)
    {
        std::vector<int> adjFaces = adjacentfaces[i];

        vertex_T[i].setZero();
        vertex_B[i].setZero();

        for(size_t j=0; j<adjFaces.size(); j++)
        {
            vertex_T[i] += face_T[adjFaces[j]];
            vertex_B[i] += face_B[adjFaces[j]];
        }
        vertex_T[i].normalize();
        vertex_B[i].normalize();
        vertex_N[i] = vertex_B[i].cross(vertex_T[i]);

		//https://www.cnblogs.com/lookof/p/3509970.html
		//normal in object space = [T,B,N] X normal in tangent space
		//light in tangent space = [T,B,N]-1 X light in object space
        vertex_TBN[i] << vertex_T[i][0], vertex_B[i][0], vertex_N[i][0],
                         vertex_T[i][1], vertex_B[i][1], vertex_N[i][1],
                         vertex_T[i][2], vertex_B[i][2], vertex_N[i][2];
    }

	std::cout<<"	done"<<endl;
	
	return;
}

#endif
