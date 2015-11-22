//
//  ViewController.m
//  MyIOS3000FPS
//
//  Created by 易佳玥 on 15/11/16.
//  Copyright © 2015年 JiayueYi. All rights reserved.
//

#import "ViewController.h"
#import "MBProgressHUD.h"
#import "LBF.h"

#import <opencv2/imgcodecs/ios.h>
#import <opencv2/videoio/cap_ios.h>
using namespace cv;

@interface ViewController ()<CvVideoCameraDelegate>

@property (weak, nonatomic) IBOutlet UIImageView *mainImg;
@property (weak, nonatomic) IBOutlet UIButton *testBtnOne;
@property (weak, nonatomic) IBOutlet UIButton *testBtnTwo;

@property (nonatomic,strong) CvVideoCamera *videoCamera;
@property (nonatomic,assign) BOOL cameraOn;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.videoCamera = [[CvVideoCamera alloc] init];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset352x288;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 5;
    self.cameraOn = NO;
}

- (void)viewDidAppear:(BOOL)animated {
    [MBProgressHUD showHUDAddedTo:self.view animated:YES];
    self.testBtnOne.enabled = NO;
    self.testBtnTwo.enabled = NO;
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [self modelInit];
        dispatch_async(dispatch_get_main_queue(), ^{
            self.testBtnOne.enabled = YES;
            self.testBtnTwo.enabled = YES;
            [MBProgressHUD hideAllHUDsForView:self.view animated:YES];
        });
    });
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)processImage:(cv::Mat &)image{
    FaceDetectionAndAlignment(image);
    dispatch_async(dispatch_get_main_queue(), ^{
        self.mainImg.image = MatToUIImage(image);
    });
}

- (IBAction)testBtn:(id)sender {
    extern std::string imagePath;
    imagePath = [[[NSBundle mainBundle] pathForResource:@"2" ofType:@"jpg"] UTF8String];
    Mat image = imread(imagePath,1);
    if (image.empty()){
        NSLog(@"Read Image fail");
        return;
    }
    FaceDetectionAndAlignment(image);
    self.mainImg.image = MatToUIImage(image);
}

- (IBAction)testBtn2:(id)sender {
    if (self.cameraOn) {
        [self.videoCamera stop];
        self.cameraOn = NO;
        [self.testBtnTwo setTitle:@"Start" forState:UIControlStateNormal];
    }else{
        [self.videoCamera start];
        self.cameraOn = YES;
        [self.testBtnTwo setTitle:@"Stop" forState:UIControlStateNormal];
    }
}


- (void)modelInit{
    extern std::string lbfModelPath;
    extern std::string regressorPath;
    extern std::string cascadePath;
    
    NSBundle* bundle = [NSBundle mainBundle];
    lbfModelPath = [[bundle pathForResource:@"LBF" ofType:@"model"] UTF8String];
    regressorPath = [[bundle pathForResource:@"Regressor" ofType:@"model"] UTF8String];
    cascadePath = [[bundle pathForResource:@"haarcascade_frontalface_alt" ofType:@"xml"] UTF8String];
    
    //load models
    static dispatch_once_t onceToken;
    
    dispatch_once(&onceToken, ^{
        LoadModelSingletons(lbfModelPath, regressorPath, cascadePath);
    });
}

@end
