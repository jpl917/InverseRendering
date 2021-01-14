//https://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// Draw a single point
static void draw_point( Mat& img, Point2f fp, Scalar color )
{
    circle( img, fp, 2, color, CV_FILLED, CV_AA, 0 );
}

// Draw delaunay triangles
// static void draw_delaunay( Mat& img, Subdiv2D& subdiv, Scalar delaunay_color )
// {
// 
//     vector<Vec6f> triangleList;
//     subdiv.getTriangleList(triangleList);
//     vector<Point> pt(3);
//     Size size = img.size();
//     Rect rect(0,0, size.width, size.height);
// 
//     for( size_t i = 0; i < triangleList.size(); i++ )
//     {
//         Vec6f t = triangleList[i];
//         pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
//         pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
//         pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
//         
//         // Draw rectangles completely inside the image.
//         if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
//         {
//             line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
//             line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
//             line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
//         }
//     }
// }

//Draw voronoi diagram
static void draw_voronoi( Mat& img, Subdiv2D& subdiv )
{
    vector<vector<Point2f> > facets;
    vector<Point2f> centers;
    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

    vector<Point> ifacet;
    vector<vector<Point> > ifacets(1);

    for( size_t i = 0; i < facets.size(); i++ )
    {
        ifacet.resize(facets[i].size());
        for( size_t j = 0; j < facets[i].size(); j++ )
            ifacet[j] = facets[i][j];

//         Scalar color;
//         color[0] = rand() & 255;
//         color[1] = rand() & 255;
//         color[2] = rand() & 255;
//         fillConvexPoly(img, ifacet, color, 8, 0);

        ifacets[0] = ifacet;
        polylines(img, ifacets, true, Scalar(0, 0, 255), 1, CV_AA, 0);
        circle(img, centers[i], 3, Scalar(0, 0, 255), CV_FILLED, CV_AA, 0);
    }
}


int main( int argc, char** argv)
{

    // Define window names
    string win_delaunay = "Delaunay Triangulation";
    string win_voronoi = "Voronoi Diagram";
    
    // Turn on animation while drawing triangles
    bool animate = true;

    // Define colors for drawing.
    Scalar delaunay_color(255,255,255), points_color(0, 0, 255);
    
    // Read in the image.
    Mat img = imread("../maya_render/env_map/indoor.JPG", cv::IMREAD_UNCHANGED);
    
    // Keep a copy around
    Mat img_orig = img.clone();
    
    // Rectangle to be used with Subdiv2D
    Size size = img.size();
    Rect rect(0, 0, size.width, size.height);

    // Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);
    
    // Create a vector of points.
    vector<Point2f> points;

    // Read in the points from a text file
    srand((unsigned)time(0));
    for(int i=0; i<size.height; i+=size.height/5){
        for(int j=0; j<size.width; j+=size.width/10){
            int tmp_i = i + int(rand() & 100);
            int tmp_j = j + int(rand() & 100);
            if(tmp_i >= size.height) tmp_i = size.height-1;
            if(tmp_j >= size.width)  tmp_j = size.width-1;
            points.push_back(cv::Point2f(tmp_j, tmp_i));
        }
    }
    
    // Insert points into subdiv
//     for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
//     {
//         subdiv.insert(*it);
        // Show animation
//         if (animate)
//         {
//             Mat img_copy = img_orig.clone();
//             // Draw delaunay triangles
//             draw_delaunay( img_copy, subdiv, delaunay_color );
//             imshow(win_delaunay, img_copy);
//             waitKey(100);
//         }
//     }
    
    
    // Draw delaunay triangles
//    draw_delaunay( img, subdiv, delaunay_color );

    // Draw points
    for( vector<Point2f>::iterator it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);
        draw_point(img, *it, points_color);
    }
    
    // Allocate space for Voronoi Diagram
    Mat img_voronoi = img.clone();//Mat::zeros(img.rows, img.cols, CV_8UC3);
    
    // Draw Voronoi diagram
    draw_voronoi( img_voronoi, subdiv );

    // Show results.
    //imshow( win_delaunay, img);
    imshow( win_voronoi, img_voronoi);
    waitKey(0);

    return 0;
}
