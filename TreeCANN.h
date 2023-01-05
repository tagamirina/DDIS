#include <cstdio>
#include <opencv2/opencv.hpp>
#include <iostream> 
#include <fstream> 
#include <sstream>
#include <string>  
#include <cstring>
#include <streambuf> 
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>

#define HUE 360

using namespace std ;
using namespace cv ;

/**********************************************/

void computeDDIS(Mat *I, Mat *T, int *patchSize, int* approximated, int* fastDiversity, vector<vector<int>>& conversionI, vector<vector<int>>& conversionT, vector<vector<double>>& DDIS, Mat* rectDDIS) ;
void run_TreeCANN(Mat *I, Mat *T, int *patchSize, int *train_patches, vector<vector<int>>& nnf_dist_temp, vector<vector<int>>& nnf_X , vector<vector<int>>& nnf_Y, vector<int>& xySrc);
void DDIS_nnf_scan(vector<vector<int>>& nnf, Mat* sT, Mat* sI, int* h, vector<vector<double>>& DDIS, Mat* rectDDIS, vector<int>& xySrc) ;
void Plot( int mode, string& TName, string& IName, Mat *T, Mat *I, vector<int>& Tcut, string& resultName, int markCol, int markRow) ;
void extract_patch_position(Mat *A, int *patch_w, vector<int>& im_patch_pos_X, vector<int>& im_patch_pos_Y) ;
void extract_patches_random(Mat *im, int *patch_w, vector<int>& pos_X, vector<int>& pos_Y, int *num_of_train_patches, vector<int>& pos_X1, vector<int>& pos_Y1, vector<int>& xySrc) ;
void propagation_stage(Mat *A, Mat *B, int *patch_w, int *number_of_patches,
    vector<int>& A_patch_pos_X, vector<int>& A_patch_pos_Y, vector<vector<int>>& nnf_X_pad, vector<vector<int>>& nnf_Y_pad, vector<vector<int>>& nnf_dist_pad) ;
void DDIS_nnf_scan_matlab(vector<vector<int>>& nnf, Mat *sT, Mat *sI, int h, vector<vector<double>>& DDIS, vector<int>& xySrc) ;
void ComputeDDISForWindow(vector<int>& nnfw, Mat* sT, Mat* sI, int h, double *DDIS, vector<int>& xySrc) ;
void unique(vector<int>& uniqueIndices, vector<int>& chosenIndexesInNnfw, vector<int>& indTransform, vector<int>& nnfw, vector<int>& Unique_, int number_patches, int *countUp) ;
void findTargetLocation(vector<vector<double>>& DDIS, Mat *rectDDIS, int ssx, int ssy) ;
void RGBtoHs( cv::Mat *II_color, vector<vector<int>>& e ) ;