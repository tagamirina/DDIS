#include "TreeCANN.h"

void run_TreeCANN(Mat *A, Mat *B, int *patch_w, int *num_of_train_patches,
    vector<vector<int>>& nnf_dist_temp, vector<vector<int>>& nnf_X , vector<vector<int>>& nnf_Y, vector<int>& xySrc){

    // function [nnf_dist_temp nnf_X nnf_Y runtime] = run_TreeCANN (A, B, patch_w, ..., 
    // A_grid, B_grid, num_of_train_patches, num_PCA_dims, eps, num_of_ann_matches, A_win, B_win, second_phase)


    // TreeCANN - Kd-Tree Coherence Nearest Neighbor algorithm
    //
    // usage: .... = TreeCANN(A,B, [patch_w=8], [A_grid = 2], [B_grid = 2], [num_of_train_patches = 100], [num_PCA_dims=patch_w/2+3], [eps=3], [num_of_ann_matches=4], [A_win=2*A_grid+1], B_win,[second_phase=1])
    //
    // This function runs the TreeCANN ( Kd-Tree Coherence Nearest Neighbor) algorithm
    // to compute the approximate dense nearest neighbor field (correspondence) between images A and B
    // -----------------------------------------------------------------------------------------------------------------
    //
    // Inputs:
    // - - - - - - -
    // 1] A - an uint8 RGB image, the source (input) of the NN field.
    // 2] B - an uint8 RGB image, the target (template) of the NN field.
    // 3] patch_w  -  the dimension [width in pixels] of a  patch.
    // 4] A_grid - sparse grid parmeter for image A
    // 5] B_grid - sparse grid parmeter for image B 
    // 6] First phase parameters (Kd-tree)
    //     6.1] num_of_train_patches - for PCA vectors calculation
    //     6.2] num_PCA_dims - patches are reduced to
    //     6.3] eps - kd-tree approximation parameter
    //     6.4] num_of_ann_matches - returned after the kd-tree search
    // 7] Seconnd phase parameters (Propagation)
    //     7.1] A_win
    //     7.2] B_win
    //     7.3] second_phase - when = '0' propagation stage is disabled
    //
    // Outputs:
    // - - - - - - - -
    // 1] nnf_dist_temp - distance field A->B
    // 2] nnf_X  - X map 
    // 3] nnf_Y  - Y map
    // 4] runtime(i)
    //    4.1] i=1 - Patch extraction
    //    4.2] i=2 - PCA vector calculation
    //    4.3] i=3 - Dimensionality reduction
    //    4.4] i=4 - Kd-tree construction
    //    4.5] i=5 - Kd-tree search
    //    4.6] i=6 - Propagation stage

    //if (nargin < 2) ;   error('Too few inputs'); end
    //if (~isa(A,'uint8'))  || ( ~isa(B,'uint8'));
    if(typeid(A) != typeid(B)) cout << "One of the input images is not Uint8" << endl ;

    //[A_patch_pos_X  A_patch_pos_Y] = extract_patch_position(size(A), patch_w, A_grid);
    //[B_patch_pos_X  B_patch_pos_Y] = extract_patch_position(size(B), patch_w, B_grid);
    //vector<int> A_patch_pos_X, A_patch_pos_Y ;
    vector<int> B_patch_pos_X, B_patch_pos_Y ;
    //extract_patch_position( A, patch_w, A_grid, A_patch_pos_X, A_patch_pos_Y ) ;
    extract_patch_position( B, patch_w, B_patch_pos_X, B_patch_pos_Y ) ;

    //----------------Reduce the amount of information-------------------------------------//

    //vector<int> A_patch_learn_X(number_patches, 0) ;
    //vector<int> A_patch_learn_Y(number_patches, 0) ;
    vector<int> B_patch_learn_X(*num_of_train_patches, 0) ;
    vector<int> B_patch_learn_Y(*num_of_train_patches, 0) ;
    //extract_patches( A, patch_w, A_patch_pos_X, A_patch_pos_Y, num_of_train_patches, A_patch_learn_X, A_patch_learn_Y ) ;
    extract_patches_random( B, patch_w, B_patch_pos_X, B_patch_pos_Y, num_of_train_patches, B_patch_learn_X, B_patch_learn_Y, xySrc ) ;

    //---------------Patch extraction------------------------------------------------//

    std::chrono::system_clock::time_point start, end ;
    start = std::chrono::system_clock::now() ;

    //----------------Propagation stage ----------------------------------------------//
    //[nnf_X_pad nnf_Y_pad nnf_dist_pad] = propagation_stage(A, B, patch_w, A_grid, A_win, B_win, A_patch_pos_X, A_patch_pos_Y, second_phase);
    //propagation_stage(A, B, patch_w, isx * isy * 2, B_patch_pos_X, B_patch_pos_Y, nnf_X, nnf_Y, nnf_dist_temp) ; // all patches
    propagation_stage(A, B, patch_w, num_of_train_patches, B_patch_learn_X, B_patch_learn_Y, nnf_X, nnf_Y, nnf_dist_temp) ;

    //nnf_X_pad = nnf_X_pad+1;
    //nnf_Y_pad = nnf_Y_pad+1;
    end = std::chrono::system_clock::now() ;
    const double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.001 ;
    cout << "TreeCANN_propagation_stage time " << time << " [msec]" << endl ;
}


/*************************************************************************************************************/

// function [patches]  = extract_patches_for_pca(im, patch_w, pos_X, pos_Y, num_of_train_patches)
void extract_patches_random(Mat *im, int *patch_w, vector<int>& pos_X, vector<int>& pos_Y, int *num_of_train_patches, vector<int>& pos_X1, vector<int>& pos_Y1, vector<int>& xySrc){

    //rand_indx = randi(size(pos_X,2),num_of_train_patches,1);
    int psize = (im->cols / *patch_w) * (im->rows / *patch_w) ;
    vector<int> rand_indx(*num_of_train_patches, 0) ;
    random_device rd ;
    default_random_engine eng(rd()) ;
    uniform_int_distribution<int> distr(0, psize - 1) ;
    for(int j = 0;j < *num_of_train_patches;j++) rand_indx[j] = distr(eng) ;

    vector<int> rand_indy(*num_of_train_patches, 0) ;
    uniform_int_distribution<int> distr2(0, psize - 1) ;
    for(int j = 0;j < *num_of_train_patches;j++) rand_indy[j] = distr2(eng) ;

    // pos_X = pos_X(rand_indx);
    // pos_Y = pos_Y(rand_indx);
    for(int k = 0;k < *num_of_train_patches;k++){
        //cout << rand_indx[k] << " " << rand_indy[k] << endl ;
        pos_X1[k] = pos_X[rand_indx[k]] ;
        pos_Y1[k] = pos_Y[rand_indy[k]] ;
        xySrc[k] = pos_Y1[k] *im->cols + pos_X1[k] ;
        //cout << "x : " << pos_X1[k] << " y : " << pos_Y1[k] << endl ;
    }
}

/*************************************************************************************************************/
// function [im_patch_pos_X  im_patch_pos_Y] = extract_patch_position(im_size,  patch_w, im_grid_step )
void extract_patch_position(Mat *A, int *patch_w, vector<int>& im_patch_pos_X, vector<int>& im_patch_pos_Y){

    int sx, sy, psx, psy, psize ;
    sx = A->cols ;
    sy = A->rows ;
    psx = (int)sx / *patch_w ;
    psy = (int)sy / *patch_w ;
    psize = psx * psy ;

    //x = 1:im_grid_step:im_size(2) - patch_w+1;
    //y = 1:im_grid_step:im_size(1) - patch_w+1;
    //[X Y] = meshgrid(y,x);
    vector<vector<int>> X(psy, vector<int>(psx, 0)) ;
    vector<vector<int>> Y(psy, vector<int>(psx, 0)) ;
    for(int j = 0;j < sy - *patch_w + 1;j = j + *patch_w){
        for(int i = 0;i < sx - *patch_w + 1;i = i + *patch_w){
            X[j / *patch_w][i / *patch_w] = j ; // 0 3 6 
            Y[j / *patch_w][i / *patch_w] = i ; // 0 0 0
            //cout << i << " " << j << endl ;
        }
    }

    //im_patch_pos_X = reshape(Y,1,numel(Y)); // numel(Y) = psize
    //im_patch_pos_Y = reshape(X,1,numel(X)); // numel(X) = psize
    im_patch_pos_X.resize(psize, 0) ;
    im_patch_pos_Y.resize(psize, 0) ;
    for(int j = 0;j < psy;j++){
        for(int i = 0;i < psx;i++){
            im_patch_pos_X[j * psx + i] = X[j][i] ;
            im_patch_pos_Y[j * psx + i] = Y[j][i] ;
        }
    }
}