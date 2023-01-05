#include "TreeCANN.h"

//function [ DDIS, rectDDIS ] = computeDDIS(I,T,patchSize, approximated, fastDiversity, conversion)
void computeDDIS(Mat *I, Mat *T, int *patchSize, int *approximated, int *fastDiversity, 
    vector<vector<int>>& conversionI, vector<vector<int>>& conversionT, vector<vector<double>>& DDIS, Mat *rectDDIS){

    // inputs:
    //  Iorig = target image (RGB)
    //  Torig = template (RGB)
    //  patchSize [3]
    //  approximated = flags the usage of TreeCANN ANN algorithm [true]
    //  fastDiversity = flags the usage of mex file or matlab implemantation [true]
    //  conversion = a converstion function for I,T in order to change its representation, 
    //  >>> for example, convert ot HSV. [@(I) I]
    //
    // outputs:
    //  DIS - Diversity Similarity heat map.
    //  rectDIS - best match rectangle according to the heat map.
    //-----------------------------------------------------------------------
    // Copyright 2022 Rina Tagami
    //
    //  nnf - NN field W X H X 2 (x ind and y ind for each patch)
    //  sT - size of the template
    //  h - bandwidth parameter (usually 1)
    //  fastDiversity - flag using c++ code [true]
    //-----------------------------------------------------------------------

    if(*fastDiversity != 1){
        conversionI.resize(I->rows, vector<int>(I->cols, 0)) ;
        conversionT.resize(T->rows, vector<int>(T->cols, 0)) ;

        Mat II = I->clone() ;
        Mat TT = T->clone() ;
        RGBtoHs(&II, conversionI) ;
        RGBtoHs(&TT, conversionT) ;
    }

    // aproximated params for TreeCANN
    int train_patches = 10 ;

    int isx, isy, tsx, tsy, size ;
    isx = I->cols ;
    isy = I->rows ;
    tsx = T->cols ;
    tsy = T->rows ;
    size = (isx - tsx + 1) * (isy - tsy + 1) ;

    // first step - NN fiels
    vector<vector<int>> nnfApprox(train_patches, vector<int>(size, 0)) ; // zeros(size(I,1),size(I,2));
    vector<vector<int>> nnf_dist_temp(train_patches, vector<int>(size, 0)) ; // nnf_dist_temp = nnf_dist
    vector<vector<int>> nnf_X(train_patches, vector<int>(size, 0)) ;
    vector<vector<int>> nnf_Y(train_patches, vector<int>(size, 0)) ;
    vector<int> xySrc(train_patches, 0) ;
    Mat sT = cv::Mat_<uchar>(T->rows, T->cols) ; // Use only cols and rows
    Mat sI = cv::Mat_<uchar>(I->rows, I->cols) ; // Use only cols and rows

    for(int j = 0;j < T->rows;j++){
        for(int i = 0;i < T->cols;i++){
            sT.at<uchar>(j, i) = 0 ;
        }
    }

    for(int j = 0;j < I->rows;j++){
        for(int i = 0;i < I->cols;i++){
            sI.at<uchar>(j, i) = 0 ;
        }
    }

    std::chrono::system_clock::time_point start, end ;
    start = std::chrono::system_clock::now() ;

    if(*approximated == 1){

        // using TreeCANN
        run_TreeCANN(I, T, patchSize, &train_patches, nnf_dist_temp, nnf_X, nnf_Y, xySrc) ;
        //nnf_X1 = nnf_X(1:end-patchSize+1,1:end-patchSize+1);
        //nnf_Y1 = nnf_Y(1:end-patchSize+1,1:end-patchSize+1);

        // remove patchSize from end
        //nnf_dist(1:end-patchSize+1,1:end-patchSize+1) = nnf_dist_temp(1:end-patchSize+1,1:end-patchSize+1);
        //nnfApprox(1:end-patchSize+1,1:end-patchSize+1) = sub2ind(sT(1:2),nnf_Y1,nnf_X1);
        for(int j = 0;j < train_patches;j++){
            for(int i = 0;i < size;i++){
                nnfApprox[j][i] = nnf_Y[j][i] * T->cols + nnf_X[j][i] ;
            }
        }
    }else{
        cout << "Error Mode" << endl ;
        /*nnf_XYD = ENN_matching(I, T, patchSize);
        nnf_X2 = nnf_XYD(1:end-patchSize+1,1:end-patchSize+1,1)+1;
        nnf_Y2 = nnf_XYD(1:end-patchSize+1,1:end-patchSize+1,2)+1;
        nnf_dist_temp = nnf_XYD(1:end-patchSize+1,1:end-patchSize+1,3)+1;
        // remove patchSize from end
        nnf_dist(1:end-patchSize+1,1:end-patchSize+1) = nnf_dist_temp(1:end-patchSize+1,1:end-patchSize+1);
        nnfExact(1:end-patchSize+1,1:end-patchSize+1) = sub2ind(sT(1:2),nnf_Y2,nnf_X2);
        nnf=nnfExact;*/
    }

    end = std::chrono::system_clock::now() ;
    const double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 ;
    cout << "run_TreeCANN time " << time << " [msec]" << endl ;

    // second step DDIS scan the NNF
    // [DDIS, rectDDIS] = DDIS_nnf_scan(nnf, sT, h, fastDiversity);
    DDIS_nnf_scan(nnfApprox, &sT, &sI, &train_patches, DDIS, rectDDIS, xySrc) ;
}

void RGBtoHs( cv::Mat *II_color, vector<vector<int>>& e2 ){
    int i, j, isx, isy ;
    int	Imax, Imin, k ;
    
    isx = II_color->cols ;
    isy = II_color->rows ;

    for(j = 0 ; j<isy ; j++){
        for(i = 0 ; i<isx ; i++){

            if((II_color->at<cv::Vec3b>(j, i)[0] == II_color->at<cv::Vec3b>(j, i)[1]) && (II_color->at<cv::Vec3b>(j, i)[1] == II_color->at<cv::Vec3b>(j, i)[2])){
                e2[j][i] = HUE ;
            }else{
            Imax = II_color->at<cv::Vec3b>(j,i)[0] ;
            Imin = II_color->at<cv::Vec3b>(j,i)[0] ;
            for(k = 1;k < 3;k++){
                if(Imax < II_color->at<cv::Vec3b>(j,i)[k]) Imax = II_color->at<cv::Vec3b>(j,i)[k] ;
                if(Imin > II_color->at<cv::Vec3b>(j,i)[k]) Imin = II_color->at<cv::Vec3b>(j,i)[k] ;
            }

            if(Imax == II_color->at<cv::Vec3b>(j, i)[2]) e2[j][i] = (double)(II_color->at<cv::Vec3b>(j,i)[1] - II_color->at<cv::Vec3b>(j,i)[0]) / (Imax - Imin) ;
            else if(Imax == II_color->at<cv::Vec3b>(j, i)[1]) e2[j][i] = (2.0 + ((double)(II_color->at<cv::Vec3b>(j,i)[0] - II_color->at<cv::Vec3b>(j,i)[2]) / (Imax - Imin))) * 60.0 ;
            else e2[j][i] = (4.0 + ((double)(II_color->at<cv::Vec3b>(j,i)[2] - II_color->at<cv::Vec3b>(j,i)[1]) / (Imax - Imin))) * 60.0 ;
            if(e2[j][i] < 0) e2[j][i] += HUE ;
            }
        }
    }
}
