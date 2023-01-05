#include "TreeCANN.h"

//-----------------------------------------------------------------------
// Copyright 2022 Rina Tagami
// For noncommercial use only.
// Please cite the appropriate paper(s) if used in research:
// Template Matching with Deformable Diversity Similarity
// Version: 1.0, 2022-12-31
//-----------------------------------------------------------------------

int main(int argc, char *argv[]){

    int mode, patchSize, verbose, markCol, markRow ;
    markCol = markRow = 0 ;
    mode = 1 ;
    string TName, IName, TxtName, resultName, output_name, RectDDIS ;
    verbose = 0 ;

    // Check Options
    for( int idx = 1; idx < argc; idx++ ){
        if( !strcmp( argv[idx], "-tmp" )) TName = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-i" )) IName = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-txt" )) TxtName = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-res" )) resultName = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-log" )) output_name = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-RectDDIS" )) RectDDIS = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-v" )) verbose = atoi( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-mode" )) mode = atoi( argv[++idx] ) ;
    }

    patchSize = 3 ; // non-overlapping patch size used for pixel descirption

    // Load images and target location
    int approximated, fastDiversity ;
    vector<vector<int>> conversionI, conversionT ;
    vector<vector<double>> heatmap ;
    Mat rectDDIS ;

    approximated = 1 ;
    fastDiversity = 1 ;

    Mat Ts = imread(TName) ;
    cout << TName << endl ;

    Mat Is = imread(IName) ;
    cout << IName << endl ;

    vector<int> Tcut(4, 0) ;
    ifstream input(TxtName) ;
    if(mode == 1) input >> Tcut[0] >> Tcut[1] >> Tcut[2] >> Tcut[3] ; // x, y, xsize, ysize
    if(mode == 0){
        string temp, temp2 ;
        getline(input, temp) ;
        istringstream ss(temp) ;
        int i = 0 ;
        do{
            ss >> Tcut[i++] ;
        }while(getline(ss, temp2, ',')) ;
    }

    cout << TxtName << endl ;
    cout << Tcut[0] << " " << Tcut[1] << " " << Tcut[2] << " " << Tcut[3] << endl ;

    // Clipping pictures for patch size
    if ((Tcut[2] % patchSize) < (patchSize / 2)) Tcut[2] -= (Tcut[2] % patchSize) ;
    else Tcut[2] += (patchSize - Tcut[2] % patchSize) ;
    if ((Tcut[3] % patchSize) < (patchSize / 2)) Tcut[3] -= (Tcut[3] % patchSize) ;
    else Tcut[3] += (patchSize - Tcut[3] % patchSize) ;

    cout << Tcut[0] << " " << Tcut[1] << " " << Tcut[2] << " " << Tcut[3] << endl ;

    // Adjust image and template size so they are divisible by the patch size 'patchSize'
    Mat T ;
    if(mode == 0) T = Ts(Rect(Tcut[0], Tcut[1], Tcut[2], Tcut[3])) ;
    if(mode == 1) T = Ts ;
    Mat I = Is(Rect(0, 0, (Is.cols - Is.cols % patchSize), (Is.rows - Is.rows % patchSize))) ;

    // can dynamically load Template and Target image using
    cout << "T size " << T.cols << " x " << T.rows << endl ;
    cout << "Target image size " << I.cols << " x " << I.rows << endl ;
    rectDDIS = Mat_<Vec3b>(I.rows - T.rows + 1, I.cols - T.cols + 1) ;

    // run (ucomment the different methods to run them)
    //------------------------------------------------------------
    
    std::chrono::system_clock::time_point start, end ;
    start = std::chrono::system_clock::now() ;

    // core function
    computeDDIS(&I, &T , &patchSize, &approximated, &fastDiversity, conversionI, conversionT, heatmap, &rectDDIS) ;

    end = std::chrono::system_clock::now() ;
    const double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 ;
    cout << "computeDDIS time " << time << " [msec]" << endl ;

    //------------------------------------------------------------

    // search matching positions
    double max = heatmap[0][0] ;

    // search max score
    int ssx = I.cols - T.cols + 1 ;
    int ssy = I.rows - T.rows + 1 ;
    for(int i = 0; i < ssy; i++){
        for(int j = 0; j < ssx; j++){
            if(heatmap[i][j] > max){
                max = heatmap[i][j] ;
                markRow = i ;
                markCol = j ;
            }
        }
    }

    if(verbose){
        // Initialize the output iamge and .txt files
        cout << output_name << endl ;
        ofstream output(output_name) ;
        output << markRow << " " << markCol << endl ; // (j, i)
        cout << markRow << " " << markCol << endl ; // (j, i)
        output.close();
    }

    // plot results
    Plot(mode, TName, IName, &T, &I, Tcut, resultName, markCol, markRow) ;
    imwrite(RectDDIS, rectDDIS) ;

    return 0 ;
}

void Plot( int mode, string& TName, string& IName, Mat *T, Mat *I, vector<int>& Tcut, string& resultName, int markCol, int markRow){

    Mat RESR, RESG, RESB, RESR2, RESG2, RESB2 ;
    Mat OUTPUT1, OUTPUT2, OUTPUT3 ;
    Mat Is2, Ts2 ;
    int colT, rowT, colI, rowI ;

    if(mode == 1) Is2 = imread(IName, 1) ;
    if(mode == 0){
        Ts2 = imread(TName, 1) ;
        Is2 = imread(IName, 1) ;
    }

    colI = I->cols ;
    rowI = I->rows ;
    colT = T->cols ;
    rowT = T->rows ;

    RESR = cv::Mat_<uchar>(rowI, colI) ;
    RESG = cv::Mat_<uchar>(rowI, colI) ;
    RESB = cv::Mat_<uchar>(rowI, colI) ;
    RESR2 = cv::Mat_<uchar>(rowI, colI) ;
    RESG2 = cv::Mat_<uchar>(rowI, colI) ;
    RESB2 = cv::Mat_<uchar>(rowI, colI) ;

    for( int j = 0 ; j < rowI ; j++ ) {
        for( int i = 0 ; i < colI ; i++ ) {
            if(mode == 0){
                RESR.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[2] ;
                RESG.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[1] ;
                RESB.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[0] ;
                RESR2.at<uchar>(j,i) = Ts2.at<cv::Vec3b>(j,i)[2] ;
                RESG2.at<uchar>(j,i) = Ts2.at<cv::Vec3b>(j,i)[1] ;
                RESB2.at<uchar>(j,i) = Ts2.at<cv::Vec3b>(j,i)[0] ;
            }else{
                RESR.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[2] ;
                RESG.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[1] ;
                RESB.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[0] ;
                RESR2.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[2] ;
                RESG2.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[1] ;
                RESB2.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[0] ;
            }
        }
    }

    // matching results
    int si, sj, ei, ej ;
    si  = markCol ;
    sj  = markRow ;
    ei  = si + colT ;
    ej  = sj + rowT ;

    //Rect-Yellow
    cv::rectangle(RESR,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESG,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESB,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(0), 1) ;
    cv::rectangle(RESR,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESG,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESB,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(0), 1) ;

    //Rect-Blue
    cv::rectangle(RESR,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(50), 1) ;
    cv::rectangle(RESG,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(255), 1) ;
    cv::rectangle(RESB,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(0), 1) ;
    
    //Rect-Yellow
    cv::rectangle(RESR,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESG,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESB,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(0), 1) ;
    cv::rectangle(RESR,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESG,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESB,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(0), 1) ;

    vector<Mat> color_img1;
    color_img1.push_back(RESB) ;
    color_img1.push_back(RESG) ;
    color_img1.push_back(RESR) ;
    merge(color_img1, OUTPUT1) ;

    // GT
    si  = Tcut[0] ;
    sj  = Tcut[1] ;
    ei  = Tcut[0] + Tcut[2] ;
    ej  = Tcut[1] + Tcut[3] ;

    //Rect-Yellow
    cv::rectangle(RESR2,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESG2,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESB2,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(0), 1) ;
    cv::rectangle(RESR2,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESG2,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESB2,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(0), 1) ;

    //Rect-Red
    cv::rectangle(RESR2,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(0), 1) ;
    cv::rectangle(RESG2,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(0), 1) ;
    cv::rectangle(RESB2,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(255), 1) ;
    
    //Rect-Yellow
    cv::rectangle(RESR2,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESG2,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESB2,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(0), 1) ;
    cv::rectangle(RESR2,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESG2,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESB2,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(0), 1) ;

    vector<Mat> color_img2 ;
    color_img2.push_back(RESB2) ;
    color_img2.push_back(RESG2) ;
    color_img2.push_back(RESR2) ;
    merge(color_img2, OUTPUT2) ;

    hconcat(OUTPUT1, OUTPUT2, OUTPUT3) ;
    cout << resultName << endl << endl ;
    imwrite(resultName, OUTPUT3) ;
}