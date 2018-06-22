/******************************************************************************/
/*                                                                            */
/*  MODEL - Main routines for Model class                                     */
/*                                                                            */
/******************************************************************************/

#define STRICT
#include <windows.h>
#include <commctrl.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <malloc.h>
#include <new.h>
#include <float.h>

#include "convnet.rh"
#include "const.h"
#include "classes.h"
#include "extern.h"
#include "funcdefs.h"

/*
--------------------------------------------------------------------------------

   Constructor and destructor

--------------------------------------------------------------------------------
*/

Model::Model ( ARCHITECTURE *arc , int nprd , int ncls )
{
   int i, k, ilayer, nfH, nfV ;
   double *gptr ;
   char msg[256] ;

   MEMTEXT ( "Model constructor" ) ;
   ok = 1 ;
   ok_to_test = 0 ;                 // Not yet trained

   n_pred = nprd ;
   n_classes = ncls ;
   n_layers = arc->n_layers ;
   for (i=0 ; i<n_layers ; i++) {
      layer_type[i] = arc->layer_type[i] ;
      depth[i] = arc->depth[i] ;
      HalfWidH[i] = arc->HalfWidH[i] ;
      HalfWidV[i] = arc->HalfWidV[i] ;
      padH[i] = arc->padH[i] ;
      padV[i] = arc->padV[i] ;
      strideH[i] = arc->strideH[i] ;
      strideV[i] = arc->strideV[i] ;
      PoolWidH[i] = arc->PoolWidH[i] ;
      PoolWidV[i] = arc->PoolWidV[i] ;
      nfH = 2 * HalfWidH[i] + 1 ;   // Filter width
      nfV = 2 * HalfWidV[i] + 1 ;

      if (layer_type[i] == TYPE_LOCAL  ||  layer_type[i] == TYPE_CONV) {
         n_prior_weights[i] = nfH * nfV ; // Inputs, soon including bias, to each neuron in this layer
         if (i == 0) {
            height[i] = (IMAGE_rows - nfV + 2 * padV[i]) / strideV[i] + 1 ;
            width[i] =  (IMAGE_cols - nfH + 2 * padH[i]) / strideH[i] + 1 ;
            n_prior_weights[i] *= IMAGE_bands ;  // Each band has a filter
            }
         else {
            height[i] = (height[i-1] - nfV + 2 * padV[i]) / strideV[i] + 1 ;
            width[i] = (width[i-1] - nfH + 2 * padH[i]) / strideH[i] + 1 ;
            n_prior_weights[i] *= depth[i-1] ;
            }
         n_prior_weights[i] += 1 ;   // Include bias
         }

      else if (layer_type[i] == TYPE_FC) {
         height[i] = width[i] = 1 ;
         if (i == 0)
            n_prior_weights[i] = n_pred + 1 ;
         else
            n_prior_weights[i] = nhid[i-1] + 1 ;
         }

      else if (layer_type[i] == TYPE_POOLAVG  ||  layer_type[i] == TYPE_POOLMAX) {
         if (i == 0) {
            height[i] = (IMAGE_rows - PoolWidV[i]) / strideV[i] + 1 ;
            width[i] =  (IMAGE_cols - PoolWidH[i]) / strideH[i] + 1 ;
            depth[i] = IMAGE_bands ;
            }
         else {
            height[i] = (height[i-1] - PoolWidV[i]) / strideV[i] + 1 ;
            width[i] =  (width[i-1] - PoolWidH[i]) / strideH[i] + 1 ;
            depth[i] = depth[i-1] ;
            }
         n_prior_weights[i] = 0 ;
         }

      else {
         assert (1 == 2) ;
         }

      nhid[i] = height[i] * width[i] * depth[i] ;
      }

   // We did hidden layers above.  Now do output layer, which is always fully connected.
   if (n_layers == 0)
      n_prior_weights[n_layers] = n_pred + 1 ;  // Final layer, always fully connected
   else
      n_prior_weights[n_layers] = nhid[n_layers-1] + 1 ;

/*
   Initialize to NULL everything that will be allocated
*/

   weights = center_wts = best_wts = NULL ;
   gradient = this_delta = prior_delta = thr_this_delta = thr_prior_delta = NULL ;
   for (i=0 ; i<MAX_LAYERS ; i++) {
      activity[i] = NULL ;
      thr_activity[0][i] = NULL ;
      poolmax_id[i] = NULL ;
      thr_poolmax_id[0][i] = NULL ;
      }
   pred = NULL ;
   conf_scratch = NULL ;
   thresh = NULL ;
   confusion = NULL ;
   thr_output = NULL ;
   thr_gradient[0] = NULL ;


/*
   Check for invalid user parameters
*/

   for (i=0 ; i<n_layers ; i++) {
      if (height[i] <= 0  ||  width[i] <= 0  ||  depth[i] <= 0) {
         sprintf_s ( msg, "User parameters in Layer %d are invalid", i+1 ) ;
         audit ( msg ) ;
         ok = 0 ;
         goto FINISH ;
         }
      }


/*
   Find the total number of weights in all hidden layers.
   This does not include the final layer.
   We need this for allocating scratch vectors for weight optimization,
   so for allocation we include the final layer weights, even though we may not need them.
   Also find the max number of neurons in any layer, including input and output.
   This is used for allocating activations.
*/

   max_any_layer = n_pred ;        // Input layer is included in max
   if (n_classes > max_any_layer)
      max_any_layer = n_classes ;  // Output layer is included in max

   n_hid_weights = 0 ;

   for (ilayer=0 ; ilayer<n_layers ; ilayer++) {      // For each of the hidden layers
      if (nhid[ilayer] > max_any_layer)
         max_any_layer = nhid[ilayer] ;
      if (layer_type[ilayer] == TYPE_FC  ||  layer_type[ilayer] == TYPE_LOCAL)
         n_hid_weights += nhid[ilayer] * n_prior_weights[ilayer] ;  // Add in weights for this layer
      else if (layer_type[ilayer] == TYPE_CONV)
         n_hid_weights += depth[ilayer] * n_prior_weights[ilayer] ; // A convolution layer uses the same weight set for every hidden neuron on visual field
                                                                    // but a different such set in each slice
      else if (layer_type[i] == TYPE_POOLAVG  ||  layer_type[i] == TYPE_POOLMAX)
         n_hid_weights += 0 ;                                       // Just for clarity; pooling has no trainable weights
      } // For ilayer (each hidden layer)

   n_all_weights = n_hid_weights + n_classes * n_prior_weights[n_layers] ;   // Add in final layer weights


/*
   Allocate memory and set pointers to individual layers
*/

   weights = (double *) MALLOC ( n_all_weights * sizeof(double) ) ;
   center_wts = (double *) MALLOC ( n_all_weights * sizeof(double) ) ;
   best_wts = (double *) MALLOC ( n_all_weights * sizeof(double) ) ;
   gradient = (double *) MALLOC ( n_all_weights * sizeof(double) ) ;
   this_delta = (double *) MALLOC ( max_any_layer * sizeof(double) ) ;
   prior_delta = (double *) MALLOC ( max_any_layer * sizeof(double) ) ;
   pred = (double *) MALLOC ( n_cases * n_classes * sizeof(double) ) ;
   conf_scratch = (double *) MALLOC ( n_cases * sizeof(double) ) ;
   thresh = (double *) MALLOC ( 6 * n_classes * sizeof(double) ) ; // 0.0, 0.5, 0.75, 0.90, 0.95, 0.99
   confusion = (int *) MALLOC ( n_classes * n_classes * sizeof(int) ) ;

   if (weights == NULL  ||  center_wts == NULL  ||  best_wts == NULL
    || gradient == NULL  ||  this_delta == NULL  ||  prior_delta == NULL
    || pred == NULL  ||  conf_scratch == NULL  ||  thresh == NULL) {
      if (weights != NULL) {
         FREE ( weights ) ;
         weights = NULL ;
         }
      if (center_wts != NULL) {
         FREE ( center_wts ) ;
         center_wts = NULL ;
         }
      if (best_wts != NULL) {
         FREE ( best_wts ) ;
         best_wts = NULL ;
         }
      if (gradient != NULL) {
         FREE ( gradient ) ;
         gradient = NULL ;
         }
      if (this_delta != NULL) {
         FREE ( this_delta ) ;
         this_delta = NULL ;
         }
      if (prior_delta != NULL) {
         FREE ( prior_delta ) ;
         prior_delta = NULL ;
         }
      if (pred != NULL) {
         FREE ( pred ) ;
         pred = NULL ;
         }
      if (conf_scratch != NULL) {
         FREE ( conf_scratch ) ;
         conf_scratch = NULL ;
         }
      if (thresh != NULL) {
         FREE (thresh ) ;
         thresh = NULL ;
         }
      if (confusion != NULL) {
         FREE ( confusion ) ;
         confusion = NULL ;
         }
      ok = 0 ;
      goto FINISH ;
      }

   k = 0 ;

   for (ilayer=0 ; ; ilayer++) {            // For each of the hidden layers, plus the final
      layer_weights[ilayer] = weights + k ;
      layer_gradient[ilayer] = gradient + k ;
      if (ilayer >= n_layers)
         break ;
      if (layer_type[ilayer] == TYPE_FC  ||  layer_type[ilayer] == TYPE_LOCAL)
         k += nhid[ilayer] * n_prior_weights[ilayer] ;  // Add in weights for this layer
      else if (layer_type[ilayer] == TYPE_CONV)
         k += depth[ilayer] * n_prior_weights[ilayer] ; // A convolution layer uses the same weights for every hidden neuron in visible field
      else if (layer_type[i] == TYPE_POOLAVG  ||  layer_type[i] == TYPE_POOLMAX)
         k += 0 ;                                       // Just for clarity; pooling has no trainable weights
      } // For ilayer (each hidden layer)

   for (ilayer=0 ; ilayer<n_layers ; ilayer++) {
      activity[ilayer] = (double *) MALLOC ( nhid[ilayer] * sizeof(double) ) ;
      if (layer_type[ilayer] == TYPE_POOLMAX)
         poolmax_id[ilayer] = (int *) MALLOC ( nhid[ilayer] * sizeof(int) ) ;
      if (activity[ilayer] == NULL  ||  (layer_type[ilayer] == TYPE_POOLMAX  &&  poolmax_id[ilayer] == NULL)) {
         for (i=0 ; i<ilayer ; i++) {
            if (activity[ilayer] != NULL) {
               FREE ( activity[ilayer] ) ;
               activity[ilayer] = NULL ;
               }
            if (poolmax_id[ilayer] != NULL) {
               FREE ( poolmax_id[ilayer] ) ;
               poolmax_id[ilayer] = NULL ;
               }
            }
         FREE ( weights ) ;
         weights = NULL ;
         FREE ( center_wts ) ;
         center_wts = NULL ;
         FREE ( best_wts ) ;
         best_wts = NULL ;
         FREE ( gradient ) ;
         gradient = NULL ;
         FREE ( this_delta ) ;
         this_delta = NULL ;
         FREE ( prior_delta ) ;
         prior_delta = NULL ;
         FREE ( pred ) ;
         pred = NULL ;
         FREE ( conf_scratch ) ;
         conf_scratch = NULL ;
         FREE (thresh ) ;
         thresh = NULL ;
         FREE ( confusion ) ;
         confusion = NULL ;
         ok = 0 ;
         goto FINISH ;
         }
      }


/*
   The following are work areas for threaded computation.
   They are quite large, and needed only if the user is not using CUDA.
   Still, it's easiest to allocate them anyway.
   Memory is cheap and straightforward construction is valuable
   Actually, confusion uses threaded computation, so until CUDA is made available
   for confusion, we need this.
*/

   thr_output = (double *) MALLOC ( n_classes * max_threads * sizeof(double) ) ;
   thr_this_delta = (double *) MALLOC ( max_any_layer * max_threads * sizeof(double) ) ;
   thr_prior_delta = (double *) MALLOC ( max_any_layer * max_threads * sizeof(double) ) ;
   thr_gradient[0] = (double *) MALLOC ( n_all_weights * max_threads * sizeof(double) ) ;

   if (thr_output == NULL  ||  thr_this_delta == NULL  ||  thr_prior_delta == NULL  ||  thr_gradient[0] == NULL) {
      if (thr_output != NULL) {
         FREE ( thr_output ) ;
         thr_output = NULL ;
         }
      if (thr_this_delta != NULL) {
         FREE ( thr_this_delta ) ;
         thr_this_delta = NULL ;
         }
      if (thr_prior_delta != NULL) {
         FREE ( thr_prior_delta ) ;
         thr_prior_delta = NULL ;
         }
      if (thr_gradient[0] != NULL) {
         FREE ( thr_gradient[0] ) ;
         thr_gradient[0] = NULL ;
         }
      audit ( "Insufficient memory allocating thread storage for training" ) ;
      ok = 0 ;
      goto FINISH ;
      }


   for (ilayer=0 ; ilayer<n_layers ; ilayer++) {
      thr_activity[0][ilayer] = (double *) MALLOC ( max_threads * nhid[ilayer] * sizeof(double) ) ;
      if (layer_type[ilayer] == TYPE_POOLMAX)
         thr_poolmax_id[0][ilayer] = (int *) MALLOC ( max_threads * nhid[ilayer] * sizeof(int) ) ;
      if (thr_activity[0][ilayer] == NULL  ||  (layer_type[ilayer] == TYPE_POOLMAX  &&  thr_poolmax_id[0][ilayer] == NULL)) {
         for (i=0 ; i<ilayer ; i++) {
            if (thr_activity[0][ilayer] != NULL) {
               FREE ( thr_activity[0][ilayer] ) ;
               thr_activity[0][ilayer] = NULL ;
               }
            if (thr_poolmax_id[0][ilayer] != NULL) {
               FREE ( thr_poolmax_id[0][ilayer] ) ;
               thr_poolmax_id[0][ilayer] = NULL ;
               }
            }
         FREE ( thr_output ) ;
         thr_output = NULL ;
         FREE ( thr_this_delta ) ;
         thr_this_delta = NULL ;
         FREE ( thr_prior_delta ) ;
         thr_prior_delta = NULL ;
         FREE ( thr_gradient[0] ) ;
         thr_gradient[0] = NULL ;
         audit ( "Insufficient memory allocating thread storage for training" ) ;
         ok = 0 ;
         goto FINISH ;
         }

      for (i=1 ; i<max_threads ; i++) {
         thr_activity[i][ilayer] = thr_activity[0][ilayer] + i * nhid[ilayer] ;
         if (layer_type[ilayer] == TYPE_POOLMAX)
            thr_poolmax_id[i][ilayer] = thr_poolmax_id[0][ilayer] + i * nhid[ilayer] ;
         }

      } // For ilayer

   for (i=0 ; i<max_threads ; i++) {
      k = 0 ;
      gptr = thr_gradient[0] + i * n_all_weights ;
      thr_gradient[i] = gptr ;
      for (ilayer=0 ; ; ilayer++) {            // For each of the hidden layers, plus the final
         thr_layer_gradient[i][ilayer] = gptr + k ;
         if (ilayer >= n_layers)
            break ;
         if (layer_type[ilayer] == TYPE_FC  ||  layer_type[ilayer] == TYPE_LOCAL)
            k += nhid[ilayer] * n_prior_weights[ilayer] ;  // Add in weights for this layer
         else if (layer_type[ilayer] == TYPE_CONV)
            k += depth[ilayer] * n_prior_weights[ilayer] ; // A convolution layer uses the same weights for every hidden neuron in visible field
         else if (layer_type[i] == TYPE_POOLAVG  ||  layer_type[i] == TYPE_POOLMAX)
            k += 0 ;                                       // Just for clarity; pooling has no trainable weights
         } // For ilayer
      } // For i (thread)


FINISH:

   // These are set (training) and used (testing) in READ_SERIES.CPP
   class_type = TrainParams.class_type ;   // 1=split zt zero; 2=split at median; 3=split at .33 and .67 quantiles
   median = TrainParams.median ;
   quantile_33 = TrainParams.quantile_33 ;
   quantile_67 = TrainParams.quantile_67 ;

   if (! ok) {
      audit ( "" ) ;
      audit ( "ERROR... Insufficient memory or other error constructing model" ) ;
      }
}


Model::~Model ()
{
   int i ;

   MEMTEXT ( "Model destructor starting" ) ;

   if (weights != NULL)
      FREE ( weights ) ;
   if (center_wts != NULL)
      FREE ( center_wts ) ;
   if (best_wts != NULL)
      FREE ( best_wts ) ;
   for (i=0 ; i<n_layers ; i++) {
      if (activity[i] != NULL)
         FREE ( activity[i] ) ;
      if (poolmax_id[i] != NULL)
         FREE ( poolmax_id[i] ) ;
      }
   if (gradient != NULL)
      FREE ( gradient ) ;
   if (this_delta != NULL)
      FREE ( this_delta ) ;
   if (prior_delta != NULL)
      FREE ( prior_delta ) ;
   if (pred != NULL)
      FREE ( pred ) ;
   if (conf_scratch != NULL)
      FREE ( conf_scratch ) ;
   if (thresh != NULL)
      FREE ( thresh ) ;
   if (confusion != NULL)
      FREE ( confusion ) ;

   MEMTEXT ( "MODEL.CPP freeing thread work areas" ) ;
   if (thr_output != NULL) {
      FREE ( thr_output ) ;
      thr_output = NULL ;
      }
   if (thr_this_delta != NULL) {
      FREE ( thr_this_delta ) ;
      thr_this_delta = NULL ;
      }
   if (thr_prior_delta != NULL) {
      FREE ( thr_prior_delta ) ;
      thr_prior_delta = NULL ;
      }
   if (thr_gradient[0] != NULL) {
      FREE ( thr_gradient[0] ) ;
      thr_gradient[0] = NULL ;
      }
   for (i=0 ; i<n_layers ; i++) {
      if (thr_activity[0][i] != NULL) {
         FREE ( thr_activity[0][i] ) ;
         thr_activity[0][i] = NULL ;
         }
      if (thr_poolmax_id[0][i] != NULL) {
         FREE ( thr_poolmax_id[0][i] ) ;
         thr_poolmax_id[0][i] = NULL ;
         }
      } // For ilayer
   MEMTEXT ( "Model destructor ending" ) ;
}


/*
--------------------------------------------------------------------------------

   print_architecture ()

--------------------------------------------------------------------------------
*/


void Model::print_architecture ()
{
   int i, total_weights ;
   char msg[1024] ;

   MEMTEXT ( "Model::print_architecture" ) ;

   audit ( "" ) ;
   cudalog ( "" ) ;
   sprintf_s ( msg, "Input has %d rows, %d columns, and %d bands", IMAGE_rows, IMAGE_cols, IMAGE_bands ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;

   audit ( "Model architecture..." ) ;
   cudalog ( "Model architecture..." ) ;
   sprintf_s ( msg, "Model has %d layers, including fully connected output", n_layers+1 ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;

   total_weights = 0 ;
   for (i=0 ; i<=n_layers ; i++) {
      if (i == n_layers) {
         sprintf_s ( msg, "   Layer %d (output) is fully connected, with %d slices (classes)",
                     i+1, n_classes ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         sprintf_s ( msg, "     %d neurons and %d prior weights per neuron gives %d weights",
                     n_classes, n_prior_weights[i], n_classes * n_prior_weights[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         total_weights += n_classes * n_prior_weights[i] ;
         }
      else if (layer_type[i] == TYPE_FC) {
         sprintf_s ( msg, "   Layer %d is fully connected, with %d slices, each %d high and %d wide",
                     i+1, depth[i], height[i], width[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         sprintf_s ( msg, "     %d neurons and %d prior weights per neuron gives %d weights",
                     nhid[i], n_prior_weights[i], nhid[i] * n_prior_weights[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         total_weights += nhid[i] * n_prior_weights[i] ;
         }
      else if (layer_type[i] == TYPE_LOCAL) {
         sprintf_s ( msg, "   Layer %d is locally connected, with %d slices, each %d high and %d wide",
                     i+1, depth[i], height[i], width[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         sprintf_s ( msg, "     Horz half-width=%d, padding=%d, stride=%d",
                     HalfWidH[i], padH[i], strideH[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         sprintf_s ( msg, "     Vert half-width=%d, padding=%d, stride=%d",
                     HalfWidV[i], padV[i], strideV[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         sprintf_s ( msg, "     %d neurons and %d prior weights per neuron gives %d weights",
                     nhid[i], n_prior_weights[i], nhid[i] * n_prior_weights[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         total_weights += nhid[i] * n_prior_weights[i] ;
         }
      else if (layer_type[i] == TYPE_CONV) {
         sprintf_s ( msg, "   Layer %d is convolutional, with %d slices, each %d high and %d wide",
                     i+1, depth[i], height[i], width[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         sprintf_s ( msg, "     Horz half-width=%d, padding=%d, stride=%d",
                     HalfWidH[i], padH[i], strideH[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         sprintf_s ( msg, "     Vert half-width=%d, padding=%d, stride=%d",
                     HalfWidV[i], padV[i], strideV[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         sprintf_s ( msg, "     %d neurons and %d prior weights per slice gives %d weights",
                     nhid[i], n_prior_weights[i], depth[i] * n_prior_weights[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         total_weights += depth[i] * n_prior_weights[i] ;
         }
      else if (layer_type[i] == TYPE_POOLAVG) {
         sprintf_s ( msg, "   Layer %d is %d by %d pooling average, with stride %d by %d, %d high, %d wide, and %d deep",
                     i+1, PoolWidV[i], PoolWidH[i], strideV[i], strideH[i], height[i], width[i], depth[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         }
      else if (layer_type[i] == TYPE_POOLMAX) {
         sprintf_s ( msg, "   Layer %d is %d by %d pooling max, with stride %d by %d, %d high, %d wide, and %d deep",
                     i+1, PoolWidV[i], PoolWidH[i], strideV[i], strideH[i], height[i], width[i], depth[i] ) ;
         audit ( msg ) ;
         cudalog ( msg ) ;
         }
      }

   assert ( total_weights == n_all_weights ) ;
   sprintf_s ( msg, "   %d Total weights for the entire model", total_weights ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;
}


/*
--------------------------------------------------------------------------------

   print_train_params ()

--------------------------------------------------------------------------------
*/

void Model::print_train_params ()
{
   char msg[256] ;

   MEMTEXT ( "Model::print_train_params" ) ;

   audit ( "" ) ;
   cudalog ( "" ) ;
   audit ( "Training parameters..." ) ;
   cudalog ( "Training parameters..." ) ;

   sprintf_s ( msg, "   Annealing iterations = %d", TrainParams.anneal_iters ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;

   sprintf_s ( msg, "   Annealing range = %.4lf", TrainParams.anneal_rng ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;

   sprintf_s ( msg, "   Refinement max iterations = %d", TrainParams.maxits ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;

   sprintf_s ( msg, "   Refinement convergence tolerance = %.5lf", TrainParams.tol ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;

   sprintf_s ( msg, "   Weight penalty = %.6lf", TrainParams.wpen ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;

   audit ( "" ) ;
   cudalog ( "" ) ;
   audit ( "CUDA parameters" ) ;
   cudalog ( "CUDA parameters" ) ;

   sprintf_s ( msg, "   Max number of cases in a training subset = %d", TrainParams.max_batch ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;

   sprintf_s ( msg, "   Max hidden neurons in a gradient launch = %d", TrainParams.max_hid_grad ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;

   sprintf_s ( msg, "   Max convolutional gradient scratch memory = %d MB", TrainParams.max_mem_grad ) ;
   audit ( msg ) ;
   cudalog ( msg ) ;
}



/*
--------------------------------------------------------------------------------

   print_weights ()

--------------------------------------------------------------------------------
*/

void Model::print_weights ()
{
   MEMTEXT ( "Model::print_weights" ) ;
   int k, iwt, filtsize, ilayer, max_cols, idepth, iheight, iwidth, ndepth, nheight, nwidth ;
   char msg[4096], ltype[256] ; ;

   max_cols = 10 ;

   audit ( "" ) ;
   k = 0 ;

   make_progress_window ( "Printing weights" ) ;

   for (ilayer=0 ; ilayer<=n_layers ; ilayer++) {

      sprintf_s ( msg, "Layer %d of %d", ilayer+1, n_layers ) ;
      write_progress ( msg ) ;

      if (ilayer == n_layers  ||  layer_type[ilayer] == TYPE_FC)
         strcpy_s ( ltype , "Fully connected" ) ;
      else if (layer_type[ilayer] == TYPE_LOCAL)
         strcpy_s ( ltype , "Locally connected" ) ;
      else if (layer_type[ilayer] == TYPE_CONV)
         strcpy_s ( ltype , "Convolutional" ) ;
      else if (layer_type[ilayer] == TYPE_POOLAVG)
         strcpy_s ( ltype , "Average pooling" ) ;
      else if (layer_type[ilayer] == TYPE_POOLMAX)
         strcpy_s ( ltype , "Mean pooling" ) ;

      if (ilayer == n_layers) {
         ndepth = n_classes ;
         nheight = nwidth = 1 ;
         }
      else {
         ndepth = depth[ilayer] ;
         nheight = height[ilayer] ;
         nwidth = width[ilayer] ;
         }

      // This (depth, height, width) loop covers all neurons in this layer.
      // LOCAL and FC layers have a different set of n_prior_weights for each neuron.
      // CONV layers have the same weight set for each neuron in the (height,width) visual field,
      // but different weight sets for each slice.

      for (idepth=0 ; idepth<ndepth ; idepth++) {
         sprintf_s ( msg, "Depth %d of %d", idepth+1, ndepth ) ;
         write_non_progress ( msg ) ;

         for (iheight=0 ; iheight<nheight ; iheight++) {
            for (iwidth=0 ; iwidth<nwidth ; iwidth++) {

               if (ilayer == n_layers  ||  layer_type[ilayer] == TYPE_FC)  // Height=width=1; nhid=depth
                  sprintf_s ( msg, "Layer %d of %d (%s)  Slice (this neuron) %d of %d",
                              ilayer+1, n_layers+1, ltype, idepth+1, ndepth ) ;
               else if (layer_type[ilayer] == TYPE_CONV)  // We will break after iheight=iwidth=0
                  sprintf_s ( msg, "Layer %d of %d (%s)  Slice %d of %d",
                              ilayer+1, n_layers+1, ltype, idepth+1, ndepth ) ;
               else if (layer_type[ilayer] == TYPE_LOCAL)
                  sprintf_s ( msg, "Layer %d of %d (%s) Slice %d of %d  Row %d of %d  Col %d of %d",
                              ilayer+1, n_layers+1, ltype, idepth+1, ndepth, iheight+1, nheight, iwidth+1, nwidth ) ;
               else if (layer_type[ilayer] == TYPE_POOLAVG  ||  layer_type[ilayer] == TYPE_POOLMAX)
                  sprintf_s ( msg, "Layer %d of %d (%s)  %d rows by %d columns by %d slices",
                              ilayer+1, n_layers+1, ltype, height[ilayer], width[ilayer], depth[ilayer] ) ;
               else
                  strcpy_s ( msg, "ERROR!!!!!!!!!!!" ) ;
               audit ( "" ) ;
               audit ( msg ) ;

               for (iwt=0 ; iwt<n_prior_weights[ilayer] ; iwt++) {
                  assert ( k < n_all_weights ) ;
                  if (iwt == n_prior_weights[ilayer]-1)
                     sprintf_s ( msg, "    %14.6lf  BIAS", weights[k] ) ;
                  else if (ilayer == 0) {
                     if (ilayer == n_layers  ||  layer_type[ilayer] == TYPE_FC)
                        sprintf_s ( msg, "    %14.6lf  Input band %d Neuron %d",
                                    weights[k], iwt/(IMAGE_rows*IMAGE_cols)+1, iwt%(IMAGE_rows*IMAGE_cols)+1 ) ;
                     else {
                        filtsize = (2*HalfWidH[ilayer]+1) * (2*HalfWidV[ilayer]+1) ;
                        sprintf_s ( msg, "    %14.6lf  Input band %d Neuron %d",
                                    weights[k], iwt/filtsize+1, iwt%filtsize+1 ) ;
                        }
                     }
                  else
                     if (ilayer == n_layers  ||  layer_type[ilayer] == TYPE_FC)
                        sprintf_s ( msg, "    %14.6lf  Prior layer slice %d Neuron %d",
                                    weights[k], iwt/(height[ilayer-1]*width[ilayer-1])+1,
                                    iwt%(height[ilayer-1]*width[ilayer-1])+1 ) ;
                     else {
                        filtsize = (2*HalfWidH[ilayer]+1) * (2*HalfWidV[ilayer]+1) ;
                        sprintf_s ( msg, "    %14.6lf  Prior layer slice %d Neuron %d",
                                    weights[k], iwt/filtsize+1, iwt%filtsize+1 ) ;
                        }
                  audit ( msg ) ;
                  if (ilayer < n_layers  &&  (layer_type[ilayer] == TYPE_LOCAL  ||  layer_type[ilayer] == TYPE_CONV)) {
                     if ((iwt+1) % ((2*HalfWidH[ilayer]+1)*(2*HalfWidV[ilayer]+1)) == 0)
                        audit ( "        -----------------------------------" ) ; // Mark end of a filter; bias is done next
                                                               // There is one of these groups for each slice in the prior layer
                     }
                  ++k ;
                  } // For iwt

               if (ilayer < n_layers  &&  (layer_type[ilayer] == TYPE_POOLAVG  ||  layer_type[ilayer] == TYPE_POOLMAX))
                  break ;       // POOL layer has no weights
               if (ilayer < n_layers  &&  layer_type[ilayer] == TYPE_CONV)
                  break ;       // In convolutional layer, weights are the same for all neurons
               } // For iwidth

            if (ilayer < n_layers  &&  (layer_type[ilayer] == TYPE_POOLAVG  ||  layer_type[ilayer] == TYPE_POOLMAX))
               break ;       // POOL layer has no weights
            if (ilayer < n_layers  &&  layer_type[ilayer] == TYPE_CONV)
               break ;
            } // For iheight

         if (ilayer < n_layers  &&  (layer_type[ilayer] == TYPE_POOLAVG  ||  layer_type[ilayer] == TYPE_POOLMAX))
            break ;       // POOL layer has no weights

         } // For idepth
      } // For ilayer

   assert ( k == n_all_weights ) ;
   destroy_progress_window ( 1 ) ;

   audit ( "" ) ;
   audit ( "" ) ;
   sprintf_s ( msg, "Negative log likelihood = %.5lf", crit - penalty ) ;
   audit ( msg ) ;
}


/*
--------------------------------------------------------------------------------

   Wrapper routines that are called often and have different versions
   for threading, no threading, and CUDA

--------------------------------------------------------------------------------
*/

double Model::trial_error ( int istart , int istop )
{
   double ll1, ll2 ;
   char msg[256] ;
#if 1
// We must not use CUDA version if checking gradient in CONJGRAD.CPP because accuracy is too low
#if CHECK_GRAD >= 3
   return trial_error_thr ( istart , istop ) ;
#endif
   if (cuda_enable)
      return trial_error_cuda ( istart , istop ) ;
   else
      return trial_error_thr ( istart , istop ) ;
#else
   ll1 = trial_error_cuda ( istart , istop ) ;
   ll2 = trial_error_no_thr ( istart , istop ) ;
   sprintf_s ( msg, "LL CUDA vs THR: %.5lf %.5lf (%.7lf)", ll1, ll2, ll1-ll2 ) ;
   MEMTEXT ( msg ) ;
   return ll2 ; // Must use host, as CUDA is not accurate enough for check_grad() tiny increments
#endif
}

double Model::grad ( int istart , int istop )
{
   if (cuda_enable)
      return grad_cuda ( istart , istop ) ;
   else
      return grad_thr ( istart , istop ) ;
}
