#include "TreeCANN.h"

// [nnf_X_pad nnf_Y_pad nnf_dist_pad] = propagation_stage(A, B, patch_w, A_grid, A_win, B_win, A_patch_pos_X, A_patch_pos_Y, second_phase);

/*******************************************************************************/
/* mexFUNCTION - gateway routine for use with MATLAB                           */
/* Calculates Approximate Nearest Neighbor dense matching between two images   */
/*******************************************************************************/

void propagation_stage(Mat *A, Mat *B, int *patch_w, int *number_of_patches,
    vector<int>& A_patch_pos_X, vector<int>& A_patch_pos_Y, 
    vector<vector<int>>& nnf_X, vector<vector<int>>& nnf_Y, vector<vector<int>>& nnf_dist_temp){

    int Swidth = A->cols - B->cols + 1 ;
    int Twidth = B->cols - *patch_w + 1 ;
    int ssx = A->cols - B->cols + 1 ;
    int ssy = A->rows - B->rows + 1 ;
    int size = ((B->cols - *patch_w + 1) * (B->rows - *patch_w + 1)) ;

    // '1' should be substructed from the fields because of the different indexing in C (starting from '0' and not from '1')
    // Compare best position in the original image space and find min distance
    for(int j = 0;j < ssy;j++){
        for(int i = 0;i < ssx;i++){
            for(int n = 0; n < *number_of_patches; n++){
                int x = A_patch_pos_X[n] ;
                int y = A_patch_pos_Y[n] ;
                //cout << "x y" << x << " " << y << endl ;
                int min_patch_dist = 10000000 ;

                for(int nn = 0; nn < size - 1; nn = nn + 3){
                    int xx = nn % Twidth ;
                    int yy = nn / Twidth ;
                    //cout << "xx yy " <<  xx << " " << yy << endl ;
                    //cout << B->cols << " " << B->rows << endl ;

                    int patch_dist = 0 ;

                    for(int dy = 0; dy < *patch_w; ++dy){
                        for(int dx = 0; dx < *patch_w; ++dx){
                            int diff_r = A->at<Vec3b>(j + y + dy, i + x + dx)[2] - B->at<Vec3b>(yy + dy, xx + dx)[2] ;
                            int diff_g = A->at<Vec3b>(j + y + dy, i + x + dx)[1] - B->at<Vec3b>(yy + dy, xx + dx)[1] ;
                            int diff_b = A->at<Vec3b>(j + y + dy, i + x + dx)[0] - B->at<Vec3b>(yy + dy, xx + dx)[0] ;
                            patch_dist += (diff_r * diff_r + diff_g * diff_g + diff_b * diff_b) ;
                            //cout << j + y + dy << " " << i + x + dx << endl ;
                            //cout << yy + dy << " " << xx + dx << endl ;
                        }
                        if(patch_dist > min_patch_dist) break ; // early termination
                    }

                    if(patch_dist < min_patch_dist){
                        min_patch_dist = patch_dist ;
                        nnf_dist_temp[n][j * Swidth + i] = patch_dist ;
                        nnf_X[n][j * Swidth + i] = xx ;
                        nnf_Y[n][j * Swidth + i] = yy ;
                        cout << "n : " << n << " i : " << i << " j : " << j << " dist : " << patch_dist << endl ;
                        //cout << "dist : " << patch_dist << " i & j  " << i << " " << j << endl ;
                    }
                }
            }
        }
    }
}