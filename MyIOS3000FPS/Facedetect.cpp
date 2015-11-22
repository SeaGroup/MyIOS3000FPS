//
//  Facedetect.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//
#include "LBFRegressor.h"
using namespace std;
using namespace cv;
int save_count=0;
Mat detectAndDraw(Mat& img,
                   CascadeClassifier& nestedCascade, LBFRegressor& regressor,
                   double scale, bool tryflip );

Mat FaceDetectionAndAlignment(Mat& image){
    // -- 0. Get LBF model
    LBFRegressor* regressor = LBFRegressor::GetInstance();
    // -- 1. Get the cascades
    CascadeClassifier* cascade = LBFRegressor::GetCascade();
    return detectAndDraw(image,*cascade,*regressor,1.3,false);
}


Mat detectAndDraw( Mat& img,
                    CascadeClassifier& cascade,
                    LBFRegressor& regressor,
                    double scale, bool tryflip ){
    int i = 0;
    double t = 0;
    vector<Rect> faces,faces2;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    
    // --Detection
    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.2, 2, 0
        |CV_HAAR_FIND_BIGGEST_OBJECT
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(80, 80));
    if( tryflip ){
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CV_HAAR_FIND_BIGGEST_OBJECT
                                 //|CV_HAAR_DO_ROUGH_SEARCH
                                 |CV_HAAR_SCALE_IMAGE
                                 ,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++ )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    
    // --Alignment
    t =(double)cvGetTickCount();
    Mat_<double> current_shape;
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ ){
        Point center;
        Scalar color = colors[i%8];
        BoundingBox boundingbox;
        
        boundingbox.start_x = r->x*scale;
        boundingbox.start_y = r->y*scale;
        boundingbox.width   = (r->width-1)*scale;
        boundingbox.height  = (r->height-1)*scale;
        boundingbox.centroid_x = boundingbox.start_x + boundingbox.width/2.0;
        boundingbox.centroid_y = boundingbox.start_y + boundingbox.height/2.0;
        
        t =(double)cvGetTickCount();
        current_shape = regressor.Predict(gray,boundingbox,1);
        t = (double)cvGetTickCount() - t;
        printf( "alignment time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
        // draw bounding box
        rectangle(img, cvPoint(boundingbox.start_x,boundingbox.start_y),
                  cvPoint(boundingbox.start_x+boundingbox.width,boundingbox.start_y+boundingbox.height),Scalar(0,255,0), 1, 8, 0);
        // draw result :: red
        for(int i = 0;i < global_params.landmark_num;i++){
             circle(img,Point2d(current_shape(i,0),current_shape(i,1)),1,Scalar(255,255,255),-1,8,0);
        }
        
//        circle(img,Point2d(current_shape(0,0),current_shape(0,1)),3,Scalar(255,0,0),-1,8,0);
    }
    return current_shape;
}
