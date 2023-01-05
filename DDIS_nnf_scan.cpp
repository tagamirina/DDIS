#include "TreeCANN.h"

// function [DDIS, rectDDIS] = DDIS_nnf_scan(nnf, sT, h, fastDiversity)
void DDIS_nnf_scan(vector<vector<int>>& nnf, Mat *sT, Mat *sI, int *number_patches, vector<vector<double>>& DDIS, Mat *rectDDIS, vector<int>& xySrc){

    //------------------------------------------------------------------------//
    // Copyright 2022 Rina Tagami
    // DDIS of C++ and OpenCV ver.
    // nnf - NN field W X H X 2 (x ind and y ind for each patch)
    // sT - size of the template
    // h - bandwidth parameter (usually 1)
    // fastDiversity - flag using c++ code [true]
    //
    // 'DDIS' is the likelihood map
    //------------------------------------------------------------------------//

    // matlab direct translation version
    // DDIS = DDIS_nnf_scan_matlab(nnf, double(sT(1:2)),h);
    DDIS_nnf_scan_matlab(nnf, sT, sI, *number_patches, DDIS, xySrc) ;

    // find target
    //DDIS = padding(DDIS,sT(1:2));
    //rectDDIS  = findTargetLocation(DDIS,locSearchStyle,[sT(2) sT(1)], windowSizeDividor, true, padMap);
    int ssx = sI->cols - sT->cols + 1 ;
    int ssy = sI->rows - sT->rows + 1 ;
    findTargetLocation(DDIS, rectDDIS, ssx, ssy) ;
}


// Matlab impl:
// function [DDIS,DIS] = DDIS_nnf_scan_matlab(nnf,sT,h)
void DDIS_nnf_scan_matlab(vector<vector<int>>& nnf, Mat *sT, Mat *sI, int number_patches, vector<vector<double>>& DDIS, vector<int>& xySrc){

    // Expand nnf
    int mnnf, nnnf ;
    vector<int> cols(sT->cols, 0) ;
    vector<int> rows(sT->rows, 0) ;

    mnnf = sI->rows - sT->rows + 1 ; // y
    nnnf = sI->cols - sT->cols + 1 ; // x
    // [mnnf,nnnf] = size(nnf);
    // mnnf = mnnf - sT(1) + 1;
    // nnnf = nnnf - sT(2) + 1;

    // Create dis
    // rows = 0:(sT(1)-1);
    // cols = 0:(sT(2)-1);
    // DIS = zeros(mnnf,nnnf);
    // DDIS = zeros(mnnf,nnnf);
    DDIS.resize(mnnf, vector<double>(nnnf, 0.0)) ;

    // Apply fun to each neighborhood of a
    for(int j = 0;j < mnnf;j++){
        for(int i = 0;i < nnnf;i++){
            vector<int> nnfw(number_patches, 0) ; // 50
            for(int l = 0;l < number_patches;l++){
                nnfw[l] = nnf[l][j * nnnf + i] ;
            }
            // nnfw = nnf(i+rows,j+cols); // linear index // nnf value
            // [DDIS(i,j)] = ComputeDDISForWindow(nnfw,sT,h);
            // DIS(i,j) = numel(unique(nnfw));

            cout << i << " " << j << endl ;
            ComputeDDISForWindow(nnfw, sT, sI, number_patches, &DDIS[j][i], xySrc) ;
        }
    }
}


// function [DDIS]= ComputeDDISForWindow(nnfw,sT,h)
void ComputeDDISForWindow(vector<int>& nnfw, Mat* sT, Mat* sI, int number_patches, double *DDIS, vector<int>& xySrc){

    int isx ;
    isx = sI->cols - sT->cols + 1 ;

    vector<int> xDest(number_patches, 0) ;
    vector<int> yDest(number_patches, 0) ;
    vector<int> xSrc(number_patches, 0) ;
    vector<int> ySrc(number_patches, 0) ;
    vector<int> u(number_patches, 0) ;
    vector<int> v(number_patches, 0) ;
    vector<double> r(number_patches, 0.0) ;
    vector<int> Unique_(number_patches, 0) ;

    for(int j = 0;j < number_patches;j++){
        xDest[j] = nnfw[j] % isx ;
        yDest[j] = nnfw[j] / isx ;
        xSrc[j] = xySrc[j] % sT->cols ;
        ySrc[j] = xySrc[j] / sT->cols ;
    }
    for(int i = 0;i < number_patches;i++){
        u[i] = xDest[i] - xSrc[i] ;
        v[i] = yDest[i] - ySrc[i] ;
        r[i] = sqrt((u[i] * u[i]) + (v[i] * v[i])) ;
    }
    // [yDest,xDest] = ind2sub(sT, nnfw(:)) ; //-patchSize+1
    // [ySrc,xSrc] = ind2sub(sT, (1:(sT(1)*sT(2)))') ; //-patchSize+1
    // u = xDest-xSrc;
    // v = yDest-ySrc;
    // [~,r] = cart2pol(u,v);
    
    // provides sorted unique list of elements
    // nnfw = [9, 9, 4, 4]
    vector<int> uniqueIndices ; // uniqueIndices = [4, 9]
    vector<int> chosenIndexesInNnfw ; // chosenIndexesInNnfw = [3, 1]
    vector<int> indTransform(number_patches, 0) ; // indTransform = [2, 2, 1, 1]
    vector<int> diversityPerPatch(number_patches, 0) ;
    vector<int> useCountForUniqueIdx(number_patches, 0) ;
    vector<double> DIw(number_patches, 0.0) ;
    int countUp = 0 ;

    // [uniqueIndices, chosenIndexesInNnfw,indTrasform] = unique(nnfw);
    unique(uniqueIndices, chosenIndexesInNnfw, indTransform, nnfw, Unique_, number_patches, &countUp) ;

    if(countUp == 1){
        useCountForUniqueIdx[0] = number_patches ;
    }else{
        // provides a count of each element's occurrence
        // useCountForUniqueIdx = hist(nnfw(:) ,uniqueIndices)';
        for(int i = 0;i < number_patches;i++) useCountForUniqueIdx[chosenIndexesInNnfw[indTransform[i]]]++ ;
    }
    for(int i = 0;i < number_patches;i++){
        //cout << "ind : " << indTransform[i] << endl ;
        cout << "use : " << useCountForUniqueIdx[i] << endl ;
    }

    // diversityPerPatch = useCountForUniqueIdx(indTrasform);
    for(int i = 0;i < number_patches;i++){
        for(int j = 0;j < countUp;j++){
            if(nnfw[i] == uniqueIndices[indTransform[chosenIndexesInNnfw[j]]]){
                diversityPerPatch[i] = useCountForUniqueIdx[chosenIndexesInNnfw[j]] ;
                break ;
            }
        }
    }

    // DIw = exp((1-diversityPerPatch) / h); // appearance
    // DDIS = sum( DIw' / (r'+1) ); // location
    for(int i = 0;i < number_patches;i++) DIw[i] = exp(1 - diversityPerPatch[i]) ;
    for(int i = 0;i < number_patches;i++) *DDIS += (DIw[i] / (r[i] + 1.0)) ;
    cout << "DDIS : " << *DDIS << endl << endl ;
}

void unique(vector<int>& uniqueIndices, vector<int>& chosenIndexesInNnfw, vector<int>& indTransform, vector<int>& nnfw, vector<int>& Unique, int number_patches, int *countUp){

    int count = 0 ;
    int flag = 0 ;

    //-----------------------------------uniqueIndices-------------------------------------//
    Unique[0] = nnfw[0] ;
    for(int i = 0;i < number_patches;i++){
        cout << "nnfw : " << nnfw[i] << endl ;
        for(int j = 0;j <= count;j++){
            if(nnfw[i] == Unique[j]){
                flag = 1 ;
                break ;
            }
        }
        if(flag == 0){
            count++ ;
            Unique[count] = nnfw[i] ;
        }
        flag = 0 ;
    }
    count++ ;
    *countUp = count ;

    uniqueIndices.resize(count, 0) ;
    cout << "count : " << count << endl ;
    for(int i = 0;i < count;i++) uniqueIndices[i] = Unique[i] ;

    // Sort
    int tmp ;
    for(int i = 0; i < count; i++) {
        for(int j = 0; j < count - 1; j++) {
            if(uniqueIndices[j] > uniqueIndices[j + 1]) {
                tmp = uniqueIndices[j] ;
                uniqueIndices[j] = uniqueIndices[j + 1] ;
                uniqueIndices[j + 1] = tmp ;
            }
        }
    }
    for(int i = 0;i < count;i++) cout << "Indices : " << uniqueIndices[i] << endl ;

    //-----------------------------------indTransform-------------------------------------//
    for(int j = 0;j < number_patches;j++){
        for(int i = 0;i < count;i++){
            if(nnfw[j] == uniqueIndices[i]){
                indTransform[j] = i ;
                break ;
            }
        }
    }

    //-----------------------------------chosenIndexesInNnfw-------------------------------------//
    chosenIndexesInNnfw.resize(count, 0) ;
    for(int j = 0;j < count;j++){
        for(int i = 0;i < number_patches;i++){
            if(uniqueIndices[j] == nnfw[i]){
                chosenIndexesInNnfw[j] = i ;
                break ;
            }
        }
    }
}

void findTargetLocation(vector<vector<double>>& DDIS, Mat *rectDDIS, int ssx, int ssy){

    for(int j = 0;j < ssy;j++){
        for(int i = 0;i < ssx;i++){

            if(DDIS[j][i] < 0.25){
                rectDDIS->at<Vec3b>(j,i)[0] = 255 ;
                rectDDIS->at<Vec3b>(j,i)[1] = DDIS[j][i] * 255;
                rectDDIS->at<Vec3b>(j,i)[2] = 0 ;
            }
            else if(DDIS[j][i] < 0.50){
                rectDDIS->at<Vec3b>(j,i)[0] = 255 - (DDIS[j][i] * 255 - 64) * 4 ;
                rectDDIS->at<Vec3b>(j,i)[1] = 255 ;
                rectDDIS->at<Vec3b>(j,i)[2] = 0 ;
            }
            else if(DDIS[j][i] < 0.75){
                rectDDIS->at<Vec3b>(j,i)[0] = 0 ;
                rectDDIS->at<Vec3b>(j,i)[1] = 255 ;
                rectDDIS->at<Vec3b>(j,i)[2] = (DDIS[j][i] * 255 - 128) * 4 ;
            }else{
                rectDDIS->at<Vec3b>(j,i)[0] = 0 ;
                rectDDIS->at<Vec3b>(j,i)[1] = 255 - (DDIS[j][i] * 255 - 192) * 4 ;
                rectDDIS->at<Vec3b>(j,i)[2] = 255 ;
            }              
        }
    }
}