//
//  LBF.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "LBF.h"
#include "LBFRegressor.h"
using namespace std;
using namespace cv;

// parameters
Params global_params;


string lbfModelPath;
string regressorPath;
string cascadePath;

string imagePath;

void ReadGlobalParamFromFile(string path){
    cout << "Loading GlobalParam..." << endl;
    ifstream fin;
    fin.open(path);
    fin >> global_params.bagging_overlap;
    fin >> global_params.max_numtrees;
    fin >> global_params.max_depth;
    fin >> global_params.max_numthreshs;
    fin >> global_params.landmark_num;
    fin >> global_params.initial_num;
    fin >> global_params.max_numstage;
    
    for (int i = 0; i< global_params.max_numstage; i++){
        fin >> global_params.max_radio_radius[i];
    }
    
    for (int i = 0; i < global_params.max_numstage; i++){
        fin >> global_params.max_numfeats[i];
    }
    cout << "Loading GlobalParam end"<<endl;
    fin.close();
}

void LoadModelSingletons(std::string lbfPath,std::string regressorPath,std::string cascadePath){
    // initialize parameters
    ReadGlobalParamFromFile(lbfPath);
    
    // load regressor
    LBFRegressor::GetInstance();
    // load cascade
    LBFRegressor::GetCascade();
}
