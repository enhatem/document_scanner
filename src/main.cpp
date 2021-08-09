#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat preProcessing(Mat& img, Mat imgGray, Mat imgBlur, Mat imgCanny, Mat imgDil)
{

    cvtColor(img,imgGray, COLOR_BGR2GRAY); // Converting image to grayscale
    GaussianBlur(imgGray,imgBlur,Size(3,3),3,0); // Introduces Gaussian blur to the image
    Canny(imgBlur,imgCanny,25, 75); // Finds edges using the Canny algorithm (common practice is to use blurred images when performing edge detection)

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3)); // the shape is MORPH_RECT(rectangular) and the size is 5x5 (if the size is increased, it will dilate more, and if we decrease it, it will dilate less, USE ODD NUMBERS ONLY) ==> We created a kernel that we can use with dilation
    dilate(imgCanny,imgDil,kernel); // increasing the thickness of the edges (to remove the gaps from the edges in the Canny image)
    // erode(imgDil,imgErode,kernel);  // decreasing the thickness of the edges

    return imgDil;
}

vector<Point> getContours(Mat imgOriginal, const Mat& imgDil){

    vector<vector<Point>> contours; // vector which contains vectors that contain points representing each contour
    vector<Vec4i> hierarchy; // vector that contains 4 integer values

    findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // drawContours(imgOriginal, contours, -1, Scalar(250,0,255), 2); // -1 means that we are drawing all of the contours

    vector<vector<Point>> conPoly(contours.size()); // we know exactly the size of conPoly since it cannot exceed the size of the contours vector
    vector<Rect> boundRect(contours.size());        // we know exactly the size of conPoly since it cannot exceed the size of the contours vector
    string objectType;

    vector<Point> biggest; // vector that will contain the points of the biggest contour in the image
    double maxArea = 0; // variable that will be updated each time a contour with a larger area is found
    // filter out noise and finding the shape of each contour
    for (int i = 0; i< contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        cout << area << endl;
        if (area > 1000) // filtering the noise
        {
            // approximating the number of curves that the contour i has.
            double peri = arcLength(contours[i],true); // calculating the perimeter using the contours and the boolean "true" is indicating that the contour is closed
            approxPolyDP(contours[i], conPoly[i], 0.02*peri, true); // finding the approximation of the curves
                                                                    // 0.02 is just a random number (we could have changed 0.02*peri to a fixed number instead)

            if (area > maxArea && conPoly[i].size() == 4) // If the contour is larger than the biggest contour found earlier, the vector that contains the contour points will be updated as well as the maxArea variable
            {
                // drawContours(imgOriginal, conPoly, i, Scalar(250,0,255), 5);    // Draw only the contours that have areas larger than 1000 to filter out the noise

                biggest = {conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3]};
                maxArea = area;
            }

            // drawContours(imgOriginal, conPoly, i, Scalar(250,0,255), 5);    // Draw only the contours that have areas larger than 1000 to filter out the noise
            // rectangle(imgOriginal, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5); // Drawing the bounding box using the rectangle - boundRect[i].tl(): calls the top left point - boundRect[i].br(): calls the bottom right point

        }
    }
    return biggest;
}

void drawPoints (Mat imgOriginal, vector<Point> points, Scalar color)
{
    for (int i = 0; i<points.size(); i++)
    {
        circle(imgOriginal,points[i],10, color, FILLED);
        putText(imgOriginal,to_string(i),points[i],FONT_HERSHEY_PLAIN,4,color,4);
    }
}

vector<Point> reorder(vector<Point> points)
{
    vector<Point> newPoints;
    vector<int> sumPoints, subPoints; // used to reorder the points that define the document corners

    for (int i=0; i< 4; i++) {
        sumPoints.push_back(points[i].x + points[i].y);
        subPoints.push_back(points[i].x - points[i].y);
    }
    newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // index 0. The subtraction will allow us to find the index of the minimum element
    newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // index 1. The subtraction will allow us to find the index of the minimum element
    newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); // index 2. The subtraction will allow us to find the index of the minimum element
    newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); // index 3. The subtraction will allow us to find the index of the minimum element

    return newPoints;
}

Mat getWarp(Mat img, Mat imgWarp, vector<Point> points, float w, float h)
{
    Point2f src[4] = {points[0], points[1], points[2], points[3] };  // the function we are going to use requires floating points. This is why Point2f was used.
    Point2f dst[4] = {{0.0f,0.0f}, {w, 0.0f}, {0.0f, h}, {w,h}}; // destination points

    Mat matrix = getPerspectiveTransform(src,dst);
    warpPerspective(img,imgWarp,matrix,Point(w,h));
    return imgWarp;
}

int main(){

    Mat imgOriginal, imgGray, imgCanny, imgThre, imgBlur, imgDil, imgErode, imgWarp, imgCrop;
    vector<Point> initialPoints, docPoints;

    float w = 420, h = 596; // Dimensions of an A4 paper multiplied by 2

    string path = "/home/elie/tutorials/c++/openCV/document_scanner/Resources/paper.jpg";
    imgOriginal = imread(path); //Mat is a matrix data-type introduced by openCV to handle images
    // resize(imgOriginal, imgOriginal,Size(), 0.5, 0.5);

    // Preprocessing
    imgThre = preProcessing(imgOriginal, imgGray, imgBlur, imgCanny, imgDil);

    // Get Contours - We will take the biggest one (the A4 paper is assumed to be the biggest rectangle in the image)
    initialPoints = getContours(imgOriginal, imgThre);
    // drawPoints(imgOriginal, initialPoints, Scalar(0,0,255));
    docPoints = reorder(initialPoints);
    // drawPoints(imgOriginal, docPoints, Scalar(0,255,0));

    // Warp
    imgWarp = getWarp(imgOriginal, imgWarp, docPoints,w,h);

    // Crop
    int cropVal = 10;
    Rect roi(cropVal,cropVal, w-(2*cropVal), h-(2*cropVal)); // defining the exact value to be cropped (cropping 5 pixels from each side)
    imgCrop = imgWarp(roi);

    imshow("Image Original",imgOriginal);
    imshow("Image Dilation",imgThre);
    imshow("Image Warp",imgWarp);
    imshow("Image Crop",imgCrop);

    waitKey(0);
    return 0;
}


//////////////////// IMPORTING VIDEOS ////////////////////
// int main(){
//     string path = "Resources/test_video.mp4";
//     VideoCapture cap(path);
//     Mat img;

//     while(true){

//         cap.read(img);
//         imshow("Image",img);
//         waitKey(20);    
//     }
//     return 0;
// }