/******************************************************************************/
/*                                                                            */
/*  MOD_THR - Assorted model routines computed on host with threading         */
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
#include <process.h>

#include "convnet.rh"
#include "const.h"
#include "classes.h"
#include "extern.h"
#include "funcdefs.h"

/*
--------------------------------------------------------------------------------

   activity_local - Compute the activation of a LOCAL layer

--------------------------------------------------------------------------------
*/

static void activity_local_thr (
   int ilayer ,                   // Layer being computed
   double *input ,                // Model inputs, used only if ilayer=0, else ignored
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   double *activity[MAX_LAYERS] , // Activity vector for each layer, used only when ilayer>0
   int HalfWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,     // And vertical
   int padH[MAX_LAYERS] ,         // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,         // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   double *layer_weights[MAX_LAYERS+1] , // Pointers to each layer's weights in 'weight' vector
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   int n_prior_weights[MAX_LAYERS+1]  // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   )
{
   int k, in_row, in_rows, in_col, in_cols, in_slice, in_slices, iheight, iwidth, idepth ;
   int rstart, rstop, cstart, cstop ;
   double sum, *wtptr, *inptr, *outptr, x ;

   assert (ilayer != n_layers) ;     // Output layer is always fully connected

   if (ilayer == 0) {
      in_rows = IMAGE_rows ;         // These are global
      in_cols = IMAGE_cols ;
      in_slices = IMAGE_bands ;
      inptr = input ;
      }
   else {
      in_rows = height[ilayer-1] ;
      in_cols = width[ilayer-1] ;
      in_slices = depth[ilayer-1] ;
      inptr = activity[ilayer-1] ;
      }

   wtptr = layer_weights[ilayer] ;   // Weights for this layer
   outptr = activity[ilayer] ;

   k = 0 ;
   for (idepth=0 ; idepth<depth[ilayer] ; idepth++) {

      for (iheight=0 ; iheight<height[ilayer] ; iheight++) {
         for (iwidth=0 ; iwidth<width[ilayer] ; iwidth++) {
            // Compute activation of this layer's neuron at (idepth, iheight, iwidth)

            sum = 0.0 ;
            // Center of first filter is at HalfWidth-Pad; filter begins at -Pad.
            rstart = strideV[ilayer] * iheight - padV[ilayer] ;
            rstop = rstart + 2 * HalfWidV[ilayer] ;
            cstart = strideH[ilayer] * iwidth - padH[ilayer] ;
            cstop = cstart + 2 * HalfWidH[ilayer] ;

            for (in_slice=0 ; in_slice<in_slices ; in_slice++) {
               for (in_row=rstart ; in_row<=rstop ; in_row++) {
                  for (in_col=cstart ; in_col<=cstop ; in_col++) {

                     // This logic is a bit inefficient
                     if (in_row >= 0  &&  in_row < in_rows  &&  in_col >= 0  &&  in_col < in_cols)
                        x = inptr[(in_slice*in_rows+in_row)*in_cols+in_col] ;
                     else
                        x = 0.0 ;
                     sum += x * *wtptr++ ;
                     } // For in_col
                  } // For in_row
               } // For in_slice
            sum += *wtptr++ ;               // Bias
            sum = exp ( 2.0 * sum ) ;
            sum = (sum - 1.0) / (sum + 1.0) ;
            outptr[k++] = sum ;
            } // For iwidth
         } // For iheight
      } // For idepth

   assert ( k == nhid[ilayer] ) ;
   assert ( layer_weights[ilayer] + nhid[ilayer] * n_prior_weights[ilayer]  == wtptr ) ;
}


/*
--------------------------------------------------------------------------------


   activity_conv - Compute the activation of a CONV layer

--------------------------------------------------------------------------------
*/

static void activity_conv_thr (
   int ilayer ,                   // Layer being computed
   double *input ,                // Model inputs, used only if ilayer=0, else ignored
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   double *activity[MAX_LAYERS] , // Activity vector for each layer, used only when ilayer>0
   int HalfWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,     // And vertical
   int padH[MAX_LAYERS] ,         // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,         // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   double *layer_weights[MAX_LAYERS+1] , // Pointers to each layer's weights in 'weight' vector
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   int n_prior_weights[MAX_LAYERS+1]  // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   )

{
   int k, in_row, in_rows, in_col, in_cols, in_slice, in_slices, iheight, iwidth, idepth ;
   int rstart, rstop, cstart, cstop ;
   double sum, *wtptr, *inptr, *outptr, x ;

   assert (ilayer != n_layers) ;     // Output layer is always fully connected

   if (ilayer == 0) {
      in_rows = IMAGE_rows ;         // These are global
      in_cols = IMAGE_cols ;
      in_slices = IMAGE_bands ;
      inptr = input ;
      }
   else {
      in_rows = height[ilayer-1] ;
      in_cols = width[ilayer-1] ;
      in_slices = depth[ilayer-1] ;
      inptr = activity[ilayer-1] ;
      }


   outptr = activity[ilayer] ;

   k = 0 ;
   wtptr = NULL ;   // Not needed; shuts up LINT

   for (idepth=0 ; idepth<depth[ilayer] ; idepth++) {

      for (iheight=0 ; iheight<height[ilayer] ; iheight++) {
         for (iwidth=0 ; iwidth<width[ilayer] ; iwidth++) {
            // Compute activation of this layer's neuron at (idepth, iheight, iwidth)
            // The weights for this layer are the same for all neurons in the layer's visual field
            // but a different such set is used for each slice

            wtptr = layer_weights[ilayer] + idepth * n_prior_weights[ilayer] ;

            sum = 0.0 ;
            // Center of first filter is at HalfWidth-Pad; filter begins at -Pad.
            rstart = strideV[ilayer] * iheight - padV[ilayer] ;
            rstop = rstart + 2 * HalfWidV[ilayer] ;
            cstart = strideH[ilayer] * iwidth - padH[ilayer] ;
            cstop = cstart + 2 * HalfWidH[ilayer] ;

            for (in_slice=0 ; in_slice<in_slices ; in_slice++) {
               for (in_row=rstart ; in_row<=rstop ; in_row++) {
                  for (in_col=cstart ; in_col<=cstop ; in_col++) {

                     // This logic is a bit inefficient
                     if (in_row >= 0  &&  in_row < in_rows  &&  in_col >= 0  &&  in_col < in_cols)
                        x = inptr[(in_slice*in_rows+in_row)*in_cols+in_col] ;
                     else
                        x = 0.0 ;
                     sum += x * *wtptr++ ;
                     } // For in_col
                  } // For in_row
               } // For in_slice
            sum += *wtptr++ ;               // Bias
            sum = exp ( 2.0 * sum ) ;
            sum = (sum - 1.0) / (sum + 1.0) ;
            outptr[k++] = sum ;
            } // For iwidth
         } // For iheight
      } // For idepth

   assert ( k == nhid[ilayer] ) ;
   assert ( layer_weights[ilayer] + depth[ilayer] * n_prior_weights[ilayer]  == wtptr ) ;
}


/*
--------------------------------------------------------------------------------

   activity_fc - Compute the activation of an FC layer

--------------------------------------------------------------------------------
*/

static void activity_fc_thr (
   int nonlin ,                   // Apply nonlinear activation function to output?
   int ilayer ,                   // Layer being computed
   double *input ,                // Model inputs, used only if ilayer=0, else ignored
   double *output ,               // Put the computed outputs here
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   double *activity[MAX_LAYERS] , // Activity vector for each layer, used only when ilayer>0
   double *layer_weights[MAX_LAYERS+1] , // Pointers to each layer's weights in 'weight' vector
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   int n_prior_weights[MAX_LAYERS+1]  // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   )

{
   int iin, iout, nin, nout ;
   double sum, *wtptr, *inptr, *outptr ;

   wtptr = layer_weights[ilayer] ;   // Weights for this layer

   if (ilayer == 0) {
      nin = n_pred ;       // This is global
      inptr = input ;
      }
   else {
      nin = nhid[ilayer-1] ;
      inptr = activity[ilayer-1] ;
      }

   assert ( nin+1 == n_prior_weights[ilayer] ) ;

   if (ilayer == n_layers) {
      nout = n_classes ;
      outptr = output ;
      }
   else {
      nout = nhid[ilayer] ;
      outptr = activity[ilayer] ;
      }

   for (iout=0 ; iout<nout ; iout++) {
      sum = 0.0 ;
      for (iin=0 ; iin<nin ; iin++)
         sum += inptr[iin] * *wtptr++ ;
      sum += *wtptr++ ;               // Bias
      if (nonlin) {
         sum = exp ( 2.0 * sum ) ;
         sum = (sum - 1.0) / (sum + 1.0) ;
         }
      outptr[iout] = sum ;
      }
}


/*
--------------------------------------------------------------------------------

   activity_pool - Compute the activation of a POOLAVG or POOLMAX layer

--------------------------------------------------------------------------------
*/

static void activity_pool_thr (
   int ilayer ,                   // Layer being computed
   double *input ,                // Model inputs, used only if ilayer=0, else ignored
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   int layer_type[MAX_LAYERS] ,   // Each entry (input to final) is TYPE_? in CONST.H
   double *activity[MAX_LAYERS] , // Activity vector for each layer, used only when ilayer>0
   int PoolWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ,     // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   int *poolmax_id[MAX_LAYERS] ,  // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int nhid[MAX_LAYERS]           // Total number of neurons in this layer = height times width times depth
   )

{
   int k, in_row, in_rows, in_col, in_cols, in_slices, iheight, iwidth, idepth ;
   int pwH, pwV, strH, strV, rstart, rstop, cstart, cstop ;
   double value, *inptr, *outptr, x ;

   assert (ilayer != n_layers) ;     // Output layer is always fully connected

   pwH = PoolWidH[ilayer] ;   // Pooling width
   pwV = PoolWidV[ilayer] ;
   strH = strideH[ilayer] ;   // Stride
   strV = strideV[ilayer] ;

   if (ilayer == 0) {
      in_rows = IMAGE_rows ;         // These are global
      in_cols = IMAGE_cols ;
      in_slices = IMAGE_bands ;
      inptr = input ;
      }
   else {
      in_rows = height[ilayer-1] ;
      in_cols = width[ilayer-1] ;
      in_slices = depth[ilayer-1] ;
      inptr = activity[ilayer-1] ;
      }

   outptr = activity[ilayer] ;

   k = 0 ;
   for (idepth=0 ; idepth<depth[ilayer] ; idepth++) {

      for (iheight=0 ; iheight<height[ilayer] ; iheight++) {
         for (iwidth=0 ; iwidth<width[ilayer] ; iwidth++) {
            // Compute activation of this layer's neuron at (idepth, iheight, iwidth)

            rstart = strV * iheight ;
            rstop = rstart + pwV - 1 ;
            cstart = strH * iwidth ;
            cstop = cstart + pwH - 1 ;

            assert ( rstop < in_rows ) ;
            assert ( cstop < in_cols ) ;

            if (layer_type[ilayer] == TYPE_POOLAVG) {
               value = 0.0 ;
               for (in_row=rstart ; in_row<=rstop ; in_row++) {
                  for (in_col=cstart ; in_col<=cstop ; in_col++)
                     value += inptr[(idepth*in_rows+in_row)*in_cols+in_col] ;
                  } // For in_row
               value /= pwV * pwH ;
               }

            else if (layer_type[ilayer] == TYPE_POOLMAX) {
               value = -1.e60 ;
               for (in_row=rstart ; in_row<=rstop ; in_row++) {
                  for (in_col=cstart ; in_col<=cstop ; in_col++) {
                     x = inptr[(idepth*in_rows+in_row)*in_cols+in_col] ;
                     if (x > value) {
                        value = x ;
                        poolmax_id[ilayer][k] = in_row * in_cols + in_col ;  // Save id of max for backprop pass
                        }
                     } // For in_col
                  } // For in_row
               }

            else
               value = 0.0 ;   // Not needed; shuts up LINT

            outptr[k++] = value ;
            } // For iwidth
         } // For iheight
      } // For idepth

   assert ( k == nhid[ilayer] ) ;
}


/*
--------------------------------------------------------------------------------

   trial - Compute the output for a given input by evaluating network

--------------------------------------------------------------------------------
*/

static void trial_thr (
   double *input ,                // Model inputs, used only if ilayer=0, else ignored
   double *output ,               // Put the computed outputs here
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   int layer_type[MAX_LAYERS] ,   // Each entry (input to final) is TYPE_? in CONST.H
   double *activity[MAX_LAYERS] , // Activity vector for each layer, used only when ilayer>0
   int HalfWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,     // And vertical
   int padH[MAX_LAYERS] ,         // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,         // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   int PoolWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ,     // And vertical
   int *poolmax_id[MAX_LAYERS] ,  // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   double *layer_weights[MAX_LAYERS+1] , // Pointers to each layer's weights in 'weight' vector
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   int n_prior_weights[MAX_LAYERS+1]  // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   )
{
   int i, ilayer ;
   double sum ;

   for (ilayer=0 ; ilayer<n_layers ; ilayer++) {     // These do not include final layer
      if (layer_type[ilayer] == TYPE_LOCAL)
         activity_local_thr ( ilayer , input , n_layers , activity , HalfWidH , HalfWidV , 
                              padH , padV , strideH , strideV , layer_weights , 
                              height , width , depth , nhid , n_prior_weights ) ;

      else if (layer_type[ilayer] == TYPE_CONV)
         activity_conv_thr ( ilayer , input , n_layers , activity , HalfWidH , HalfWidV , 
                             padH , padV , strideH , strideV , layer_weights , 
                             height , width , depth , nhid , n_prior_weights ) ;

      else if (layer_type[ilayer] == TYPE_FC)
         activity_fc_thr ( 1 , ilayer , input , output , n_layers , activity , 
                           layer_weights , nhid , n_prior_weights ) ;

      else if (layer_type[ilayer] == TYPE_POOLAVG  ||  layer_type[ilayer] == TYPE_POOLMAX)
         activity_pool_thr ( ilayer , input , n_layers , layer_type , activity , 
                             PoolWidH , PoolWidV , strideH , strideV , poolmax_id , 
                             height , width , depth , nhid ) ;

      else
         assert ( 1 == 2 ) ;
      }
         
   activity_fc_thr ( 0 , ilayer , input , output , n_layers , activity , 
                     layer_weights , nhid , n_prior_weights ) ;

   // Classifier is always SoftMax
   sum = 1.e-60 ;
   for (i=0 ; i<n_classes ; i++) {
      if (output[i] < 300.0)
         output[i] = exp ( output[i] ) ;
      else
         output[i] = exp ( 300.0 ) ;
      sum += output[i] ;
      }
   for (i=0 ; i<n_classes ; i++)
      output[i] /= sum ;
}


/*
--------------------------------------------------------------------------------

   compute_nonpooled_delta - Compute all deltas from any layer type to a LOCAL or CONV next layer

--------------------------------------------------------------------------------
*/

static void compute_nonpooled_delta (
   int ilayer ,                   // Layer being computed
   int layer_type[MAX_LAYERS] ,   // Each entry (input to final) is TYPE_? in CONST.H
   int HalfWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,     // And vertical
   int padH[MAX_LAYERS] ,         // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,         // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   double *layer_weights[MAX_LAYERS+1] , // Pointers to each layer's weights in 'weight' vector
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   double *this_delta ,           // Scratch vector for gradient computation
   double *prior_delta ,          // Ditto
   int n_prior_weights[MAX_LAYERS+1]  // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
)
{
   int i, hwH, nH, hwV, nV, pdH, pdV, next_row, next_col, next_slice, next_rows, next_cols, next_slices ;
   int this_slices, this_rows, this_cols, idepth, iheight, iwidth ;
   int rstart, rstop, cstart, cstop, strH, strV, k_this, k_next ;
   double *wtptr ;

   for (i=0 ; i<nhid[ilayer] ; i++)
      prior_delta[i] = 0.0 ;

   hwH = HalfWidH[ilayer+1] ;  // Filter half-width in next layer
   nH = 2 * hwH + 1 ;          // And its number of columns
   hwV = HalfWidV[ilayer+1] ;  // Ditto for rows
   nV = 2 * hwV + 1 ;
   strH = strideH[ilayer+1] ;
   strV = strideV[ilayer+1] ;
   pdH = padH[ilayer+1] ;
   pdV = padV[ilayer+1] ;

   this_rows = height[ilayer] ;
   this_cols = width[ilayer] ;
   this_slices = depth[ilayer] ;

   next_rows = height[ilayer+1] ;
   next_cols = width[ilayer+1] ;
   next_slices = depth[ilayer+1] ;

/*
   Loop through every possible connection from a neuron in ilayer to a neuron in the next layer
*/

   k_next= 0 ;  // Will index every neuron in the next layer
   for (next_slice=0 ; next_slice<next_slices ; next_slice++) {
      for (next_row=0 ; next_row<next_rows ; next_row++) {
         for (next_col=0 ; next_col<next_cols ; next_col++) {

            if (layer_type[ilayer+1] == TYPE_CONV)       // A CONV layer has the same weight set for all neurons in visible field
               wtptr = layer_weights[ilayer+1] + next_slice * n_prior_weights[ilayer+1] ;
            else if (layer_type[ilayer+1] == TYPE_LOCAL) // A LOCAL layer has different weights for each neuron
               wtptr = layer_weights[ilayer+1] + k_next * n_prior_weights[ilayer+1] ; // Weights for neuron k_next
            else
               wtptr = NULL ;    // Not needed.  Shuts up LINT

            // Center of first filter is at HalfWidth-Pad; filter begins at -Pad.
            rstart = strV * next_row - pdV ;
            rstop = rstart + 2 * hwV ;
            cstart = strH * next_col - pdH ;
            cstop = cstart + 2 * hwH ;

            for (idepth=0 ; idepth<this_slices ; idepth++) {
               for (iheight=rstart ; iheight<=rstop ; iheight++) {
                  for (iwidth=cstart ; iwidth<=cstop ; iwidth++) {

                     if (iheight >= 0  &&  iheight < this_rows  &&  iwidth >= 0  &&  iwidth < this_cols) {
                        k_this = (idepth * this_rows + iheight) * this_cols + iwidth ;
                        assert ( k_this < nhid[ilayer] ) ;
                        prior_delta[k_this] += this_delta[k_next] * *wtptr++ ;
                        }
                     else
                        ++wtptr ;
                     }  // For iwidth
                  }  // For iheight
               }  // For idepth
            ++k_next ;
            }  // For next_col
         }  // For next_row
      }  // For next_slice

   assert ( k_next == nhid[ilayer+1] ) ;
}


/*
--------------------------------------------------------------------------------

   compute_pooled_delta - Compute all deltas from any layer type
                          to a POOLEDAVG or POOLEDMAX next layer

--------------------------------------------------------------------------------
*/

static void compute_pooled_delta (
   int ilayer ,                   // Layer being computed
   int layer_type[MAX_LAYERS] ,   // Each entry (input to final) is TYPE_? in CONST.H
   int PoolWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ,     // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   double *this_delta ,           // Scratch vector for gradient computation
   double *prior_delta ,          // Ditto
   int *poolmax_id[MAX_LAYERS]    // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
)
{
   int i, pwH, pwV, next_row, next_col, next_slice, next_rows, next_cols, next_slices ;
   int this_slices, this_rows, this_cols, iheight, iwidth ;
   int rstart, rstop, cstart, cstop, strH, strV, k_this, k_next ;
   double wt ;

   for (i=0 ; i<nhid[ilayer] ; i++)
      prior_delta[i] = 0.0 ;

   pwH = PoolWidH[ilayer+1] ;  // Pooling filter width in next layer
   pwV = PoolWidV[ilayer+1] ;
   strH = strideH[ilayer+1] ;
   strV = strideV[ilayer+1] ;

   this_rows = height[ilayer] ;
   this_cols = width[ilayer] ;
   this_slices = depth[ilayer] ;

   next_rows = height[ilayer+1] ;
   next_cols = width[ilayer+1] ;
   next_slices = depth[ilayer+1] ;

/*
   Loop through every possible connection from a neuron in ilayer to a neuron in the next layer
*/

   k_next= 0 ;  // Will index every neuron in the next layer
   for (next_slice=0 ; next_slice<next_slices ; next_slice++) {
      for (next_row=0 ; next_row<next_rows ; next_row++) {
         for (next_col=0 ; next_col<next_cols ; next_col++) {

            if (layer_type[ilayer+1] == TYPE_POOLAVG){
               wt = 1.0 / (pwH * pwV) ;
               rstart = strV * next_row ;
               rstop = rstart + pwV - 1 ;
               cstart = strH * next_col ;
               cstop = cstart + pwH - 1 ;
   
               for (iheight=rstart ; iheight<=rstop ; iheight++) {
                  for (iwidth=cstart ; iwidth<=cstop ; iwidth++) {
                     k_this = (next_slice * this_rows + iheight) * this_cols + iwidth ;
                     assert ( k_this < nhid[ilayer] ) ;
                     prior_delta[k_this] += this_delta[k_next] * wt ;
                     }  // For iwidth
                  }  // For iheight
               }  // If POOLAVG

            else if (layer_type[ilayer+1] == TYPE_POOLMAX) {
               iheight = poolmax_id[ilayer+1][k_next] / this_cols ;
               iwidth = poolmax_id[ilayer+1][k_next] % this_cols ;
               k_this = (next_slice * this_rows + iheight) * this_cols + iwidth ;
               assert ( k_this < nhid[ilayer] ) ;
               prior_delta[k_this] += this_delta[k_next] ;  // Weight is 1
               }

            ++k_next ;
            }  // For next_col
         }  // For next_row
      }  // For next_slice

   assert ( k_next == nhid[ilayer+1] ) ;
}


/*
--------------------------------------------------------------------------------

   grad_no_thr_FC - Gradient for a fully connected layer

--------------------------------------------------------------------------------
*/

static void grad_thr_FC (
   int icase ,                    // Case being computed
   int ilayer ,                   // Layer being computed
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   int layer_type[MAX_LAYERS] ,   // Each entry (input to final) is TYPE_? in CONST.H
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int HalfWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,     // And vertical
   int PoolWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ,     // And vertical
   int padH[MAX_LAYERS] ,         // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,         // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   int *poolmax_id[MAX_LAYERS] ,  // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   double *activity[MAX_LAYERS] , // Activity vector for each layer, used only when ilayer>0
   double *layer_weights[MAX_LAYERS+1] ,  // Pointers to each layer's weights in 'weight' vector
   double *layer_gradient[MAX_LAYERS+1] , // Pointers to each layer's gradient in 'gradient' vector
   double *this_delta ,           // Scratch vector for gradient computation
   double *prior_delta ,          // Ditto
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   int n_prior_weights[MAX_LAYERS+1]  // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   )
{
   int i, j, nthis, nnext ;
   double *gradptr, delta, *prevact, *nextcoefs ;

   nthis = nhid[ilayer] ;      // Number of neurons in this hidden layer (height * width * depth)
   if (ilayer == n_layers-1)   // Next layer is output layer?
      nnext = n_classes ;
   else
      nnext = nhid[ilayer+1] ;

   if (ilayer == 0) {                          // First hidden layer?
      prevact = database + icase * n_db_cols ; // Point to this sample (database is global, as is n_db_cols)
      assert ( n_prior_weights[ilayer]-1 == n_pred ) ;
      }
   else {      // There is at least one more hidden layer prior to this one
      prevact = activity[ilayer-1] ;
      assert ( n_prior_weights[ilayer]-1 == nhid[ilayer-1] ) ;
      }

   gradptr = layer_gradient[ilayer] ; // Point to gradient for this layer
   nextcoefs = layer_weights[ilayer+1] ;

   for (i=0 ; i<nthis ; i++) {        // For each neuron in this layer

      if (ilayer+1 == n_layers  ||  layer_type[ilayer+1] == TYPE_FC) { // Simple case of full connection
         delta = 0.0 ;
         for (j=0 ; j<nnext ; j++)
            delta += this_delta[j] * nextcoefs[j*(nthis+1)+i] ;
         }

      else if (i == 0) {  // Will compute all deltas at once
         if (layer_type[ilayer+1] == TYPE_LOCAL  ||  layer_type[ilayer+1] == TYPE_CONV)
            compute_nonpooled_delta ( ilayer , layer_type ,
                         HalfWidH , HalfWidV , padH , padV , strideH , strideV ,
                         layer_weights , height , width , depth , nhid ,
                         this_delta , prior_delta , n_prior_weights ) ;
         else if (layer_type[ilayer+1] == TYPE_POOLAVG  ||  layer_type[ilayer+1] == TYPE_POOLMAX)
            compute_pooled_delta ( ilayer , layer_type ,
                         PoolWidH , PoolWidV , strideH , strideV ,
                         height , width , depth , nhid ,
                         this_delta , prior_delta , poolmax_id ) ;
         delta = prior_delta[i] ;
         }

      else
         delta = prior_delta[i] ;  // It's already computed (just above) and saved

      delta *= 1.0 - activity[ilayer][i] * activity[ilayer][i] ;  // Derivative
      prior_delta[i] = delta ;                    // Save it for the next layer back

      for (j=0 ; j<n_prior_weights[ilayer]-1 ; j++)  // Don't include bias here
         *gradptr++ += delta * prevact[j] ;
      *gradptr++ += delta ;   // Bias activation is always 1
      }  // For all neurons in this hidden layer
}


/*
--------------------------------------------------------------------------------

   grad_no_thr_LOCAL - Gradient for a LOCAL layer

--------------------------------------------------------------------------------
*/

static void grad_thr_LOCAL (
   int icase ,                    // Case being computed
   int ilayer ,                   // Layer being computed
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   int layer_type[MAX_LAYERS] ,   // Each entry (input to final) is TYPE_? in CONST.H
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int HalfWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,     // And vertical
   int PoolWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ,     // And vertical
   int padH[MAX_LAYERS] ,         // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,         // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   int *poolmax_id[MAX_LAYERS] ,  // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   double *activity[MAX_LAYERS] , // Activity vector for each layer, used only when ilayer>0
   double *layer_weights[MAX_LAYERS+1] ,  // Pointers to each layer's weights in 'weight' vector
   double *layer_gradient[MAX_LAYERS+1] , // Pointers to each layer's gradient in 'gradient' vector
   double *this_delta ,           // Scratch vector for gradient computation
   double *prior_delta ,          // Ditto
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   int n_prior_weights[MAX_LAYERS+1]  // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   )
{
   int j, k, nthis, nnext, idepth, iheight, iwidth ;
   int in_row, in_col, in_slice, in_rows, in_cols, in_slices ;
   int rstart, rstop, cstart, cstop ;
   double *gradptr, delta, *prevact, *nextcoefs, x ;

   nthis = nhid[ilayer] ;      // Number of neurons in this hidden layer (height * width * depth)
   if (ilayer == n_layers-1)   // Next layer is output layer?
      nnext = n_classes ;
   else
      nnext = nhid[ilayer+1] ;

   if (ilayer == 0) {
      prevact = database + icase * n_db_cols ; // Point to this sample (database is global, as is n_db_cols)
      in_rows = IMAGE_rows ;         // These are global
      in_cols = IMAGE_cols ;
      in_slices = IMAGE_bands ;
      }
   else {
      prevact = activity[ilayer-1] ;
      in_rows = height[ilayer-1] ;
      in_cols = width[ilayer-1] ;
      in_slices = depth[ilayer-1] ;
      }

   gradptr = layer_gradient[ilayer] ;    // Point to gradient for this layer
   nextcoefs = layer_weights[ilayer+1] ; // Weights for next layer

   k = 0 ;  // This will index the nhid[ilayer] neurons in this layer
   for (idepth=0 ; idepth<depth[ilayer] ; idepth++) {

      for (iheight=0 ; iheight<height[ilayer] ; iheight++) {
         for (iwidth=0 ; iwidth<width[ilayer] ; iwidth++) {

            //-------------------------------------------------------------------------------------------
            // We are now inside the three nested loops that cover all nhid[ilayer] neurons in this layer
            // Compute delta for this neuron by summing across all connections to the next layer
            //-------------------------------------------------------------------------------------------

            if (ilayer+1 == n_layers  ||  layer_type[ilayer+1] == TYPE_FC) { // Simple case of full connection
               delta = 0.0 ;
               for (j=0 ; j<nnext ; j++)
                  delta += this_delta[j] * nextcoefs[j*(nthis+1)+k] ;
               }

            else if (idepth == 0  &&  iheight == 0  &&  iwidth == 0) {  // Will compute all deltas at once
               if (layer_type[ilayer+1] == TYPE_LOCAL  ||  layer_type[ilayer+1] == TYPE_CONV)
                  compute_nonpooled_delta ( ilayer , layer_type ,
                               HalfWidH , HalfWidV , padH , padV , strideH , strideV ,
                               layer_weights , height , width , depth , nhid ,
                               this_delta , prior_delta , n_prior_weights ) ;

               else if (layer_type[ilayer+1] == TYPE_POOLAVG  ||  layer_type[ilayer+1] == TYPE_POOLMAX)
                  compute_pooled_delta ( ilayer , layer_type ,
                               PoolWidH , PoolWidV , strideH , strideV ,
                               height , width , depth , nhid ,
                               this_delta , prior_delta , poolmax_id ) ;
               delta = prior_delta[k] ;
               }

            else
               delta = prior_delta[k] ;  // It's already computed (just above) and saved


            // At this point, delta for this layer's hidden neuron k at (idepth, iheight, iwidth)
            // is the derivative of the criterion wrt the output of this hidden neuron.
            // To make it be wrt the input to this neuron, multiply by the derivative
            // of the activation function.
            // Note that this multiplication takes place only once for each neuron k.

            delta *= 1.0 - activity[ilayer][k] * activity[ilayer][k] ;  // Derivative
            prior_delta[k] = delta ;   // Save it for the next layer back
                                       // Delta is the derivative of the crit wrt net input to neuron k

            //----------------------------------------------------------------------------------
            // To get the derivative of the criterion with respect to the n_prior_weights
            // coming into this neuron, multiply delta by the corresponding input to the weight.
            //----------------------------------------------------------------------------------

            // Center of first filter is at HalfWidth-Pad; filter begins at -Pad.
            rstart = strideV[ilayer] * iheight - padV[ilayer] ;
            rstop = rstart + 2 * HalfWidV[ilayer] ;
            cstart = strideH[ilayer] * iwidth - padH[ilayer] ;
            cstop = cstart + 2 * HalfWidH[ilayer] ;

            for (in_slice=0 ; in_slice<in_slices ; in_slice++) {
               for (in_row=rstart ; in_row<=rstop ; in_row++) {
                  for (in_col=cstart ; in_col<=cstop ; in_col++) {

                     // This logic is a bit inefficient
                     if (in_row >= 0  &&  in_row < in_rows  &&  in_col >= 0  &&  in_col < in_cols)
                        x = prevact[(in_slice*in_rows+in_row)*in_cols+in_col] ;
                     else
                        x = 0.0 ;
                     *gradptr++ += delta * x ;
                     } // For every column in the prior layer
                  } // For every row in the prior layer
               } // For every slice in the prior layer

            *gradptr++ += delta ;   // Bias activation is always 1
            ++k ;
            }  // For width dimension in this hidden layer
         }  // For height dimension in this hidden layer
      }  // For depth dimension in this hidden layer
   
   assert ( k == nthis ) ;
   assert ( layer_gradient[ilayer] + nhid[ilayer] * n_prior_weights[ilayer]  == gradptr ) ;
}


/*
--------------------------------------------------------------------------------

   grad_no_thr_CONV - Gradient for a CONV layer

--------------------------------------------------------------------------------
*/

static void grad_thr_CONV (
   int icase ,                    // Case being computed
   int ilayer ,                   // Layer being computed
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   int layer_type[MAX_LAYERS] ,   // Each entry (input to final) is TYPE_? in CONST.H
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int HalfWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,     // And vertical
   int PoolWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ,     // And vertical
   int padH[MAX_LAYERS] ,         // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,         // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   int *poolmax_id[MAX_LAYERS] ,  // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   double *activity[MAX_LAYERS] , // Activity vector for each layer, used only when ilayer>0
   double *layer_weights[MAX_LAYERS+1] ,  // Pointers to each layer's weights in 'weight' vector
   double *layer_gradient[MAX_LAYERS+1] , // Pointers to each layer's gradient in 'gradient' vector
   double *this_delta ,           // Scratch vector for gradient computation
   double *prior_delta ,          // Ditto
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   int n_prior_weights[MAX_LAYERS+1]  // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   )
{
   int j, k, nthis, nnext, idepth, iheight, iwidth ;
   int in_row, in_col, in_slice, in_rows, in_cols, in_slices ;
   int rstart, rstop, cstart, cstop ;
   double *gradptr, delta, *prevact, *nextcoefs, x ;

   nthis = nhid[ilayer] ;      // Number of neurons in this hidden layer (height * width * depth)
   if (ilayer == n_layers-1)   // Next layer is output layer?
      nnext = n_classes ;
   else
      nnext = nhid[ilayer+1] ;

   if (ilayer == 0) {
      prevact = database + icase * n_db_cols ; // Point to this sample (database is global, as is n_db_cols)
      in_rows = IMAGE_rows ;         // These are global
      in_cols = IMAGE_cols ;
      in_slices = IMAGE_bands ;
      }
   else {
      prevact = activity[ilayer-1] ;
      in_rows = height[ilayer-1] ;
      in_cols = width[ilayer-1] ;
      in_slices = depth[ilayer-1] ;
      }

   nextcoefs = layer_weights[ilayer+1] ; // Weights for next layer

   k = 0 ;  // This will index the nhid[ilayer] neurons in this layer
   for (idepth=0 ; idepth<depth[ilayer] ; idepth++) {
      for (iheight=0 ; iheight<height[ilayer] ; iheight++) {
         for (iwidth=0 ; iwidth<width[ilayer] ; iwidth++) {

            //-------------------------------------------------------------------------------------------
            // We are now inside the three nested loops that cover all nhid[ilayer] neurons in this layer
            // Compute delta for this neuron by summing across all connections to the next layer
            //-------------------------------------------------------------------------------------------

            if (ilayer+1 == n_layers  ||  layer_type[ilayer+1] == TYPE_FC) { // Simple case of full connection
               delta = 0.0 ;
               for (j=0 ; j<nnext ; j++)
                  delta += this_delta[j] * nextcoefs[j*(nthis+1)+k] ;
               }

            else if (idepth == 0  &&  iheight == 0  &&  iwidth == 0) {  // Will compute all deltas at once
               if (layer_type[ilayer+1] == TYPE_LOCAL  ||  layer_type[ilayer+1] == TYPE_CONV)
                  compute_nonpooled_delta ( ilayer , layer_type ,
                               HalfWidH , HalfWidV , padH , padV , strideH , strideV ,
                               layer_weights , height , width , depth , nhid ,
                               this_delta , prior_delta , n_prior_weights ) ;
               else if (layer_type[ilayer+1] == TYPE_POOLAVG  ||  layer_type[ilayer+1] == TYPE_POOLMAX)
                  compute_pooled_delta ( ilayer , layer_type ,
                               PoolWidH , PoolWidV ,  strideH , strideV ,
                               height , width , depth , nhid ,
                               this_delta , prior_delta , poolmax_id ) ;
               delta = prior_delta[k] ;
               }

            else
               delta = prior_delta[k] ;  // It's already computed (just above) and saved


            // At this point, delta for this layer's hidden neuron k at (idepth, iheight, iwidth)
            // is the derivative of the criterion wrt the output of this hidden neuron.
            // To make it be wrt the input to this neuron, multiply by the derivative
            // of the activation function.
            // Note that this multiplication takes place only once for each neuron k.

            delta *= 1.0 - activity[ilayer][k] * activity[ilayer][k] ;  // Derivative
            prior_delta[k] = delta ;   // Save it for the next layer back
                                       // Delta is the derivative of the crit wrt net input to neuron k

            //----------------------------------------------------------------------------------
            // To get the derivative of the criterion with respect to the n_prior_weights
            // coming into this neuron, multiply delta by the corresponding input to the weight.
            //----------------------------------------------------------------------------------

            // Weights for this layer are the same for all neurons in the visible field
            // But a different set is used for each slice in this layer

            gradptr = layer_gradient[ilayer] + idepth * n_prior_weights[ilayer] ;

            // Center of first filter is at HalfWidth-Pad; filter begins at -Pad.
            rstart = strideV[ilayer] * iheight - padV[ilayer] ;
            rstop = rstart + 2 * HalfWidV[ilayer] ;
            cstart = strideH[ilayer] * iwidth - padH[ilayer] ;
            cstop = cstart + 2 * HalfWidH[ilayer] ;

            for (in_slice=0 ; in_slice<in_slices ; in_slice++) {
               for (in_row=rstart ; in_row<=rstop ; in_row++) {
                  for (in_col=cstart ; in_col<=cstop ; in_col++) {

                     // This logic is a bit inefficient
                     if (in_row >= 0  &&  in_row < in_rows  &&  in_col >= 0  &&  in_col < in_cols)
                        x = prevact[(in_slice*in_rows+in_row)*in_cols+in_col] ;
                     else
                        x = 0.0 ;
                     *gradptr++ += delta * x ;
                     } // For every column in the prior layer
                  } // For every row in the prior layer
               } // For every slice in the prior layer

            *gradptr++ += delta ;   // Bias activation is always 1
            ++k ;
            assert ( layer_gradient[ilayer] + (idepth+1) * n_prior_weights[ilayer]  == gradptr ) ;
            }  // For width dimension in this hidden layer
         }  // For height dimension in this hidden layer
      }  // For depth dimension in this hidden layer
   
   assert ( k == nthis ) ;
}


/*
--------------------------------------------------------------------------------

   grad_no_thr_POOL - POOL layer has no gradient (!), but this progates delta

--------------------------------------------------------------------------------
*/

static void grad_thr_POOL (
   int ilayer ,                   // Layer being computed
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   int layer_type[MAX_LAYERS] ,   // Each entry (input to final) is TYPE_? in CONST.H
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   int HalfWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,     // And vertical
   int PoolWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ,     // And vertical
   int padH[MAX_LAYERS] ,         // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,         // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   int *poolmax_id[MAX_LAYERS] ,  // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   double *layer_weights[MAX_LAYERS+1] , // Pointers to each layer's weights in 'weight' vector
   int n_prior_weights[MAX_LAYERS+1] , // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   double *this_delta ,           // Scratch vector for gradient computation
   double *prior_delta            // Ditto
)
{
   int j, k,nthis, nnext, idepth, iheight, iwidth ;
   double delta, *nextcoefs ;

   nthis = nhid[ilayer] ;      // Number of neurons in this hidden layer (height * width * depth)

   if (ilayer == n_layers-1)   // Next layer is output layer?
      nnext = n_classes ;
   else
      nnext = nhid[ilayer+1] ;
   nextcoefs = layer_weights[ilayer+1] ; // Weights for next layer

   k = 0 ;  // This will index the nhid[ilayer] neurons in this layer
   for (idepth=0 ; idepth<depth[ilayer] ; idepth++) {
      for (iheight=0 ; iheight<height[ilayer] ; iheight++) {
         for (iwidth=0 ; iwidth<width[ilayer] ; iwidth++) {

            //-------------------------------------------------------------------------------------------
            // We are now inside the three nested loops that cover all nhid[ilayer] neurons in this layer
            // Compute delta for this neuron by summing across all connections to the next layer
            //-------------------------------------------------------------------------------------------

            if (ilayer+1 == n_layers  ||  layer_type[ilayer+1] == TYPE_FC) { // Simple case of full connection
               delta = 0.0 ;
               for (j=0 ; j<nnext ; j++)
                  delta += this_delta[j] * nextcoefs[j*(nthis+1)+k] ;
               prior_delta[k] = delta ;   // Save it for the next layer back
               }

            else if (idepth == 0  &&  iheight == 0  &&  iwidth == 0) {  // Will compute all deltas at once
               if (layer_type[ilayer+1] == TYPE_LOCAL  ||  layer_type[ilayer+1] == TYPE_CONV)
                  compute_nonpooled_delta ( ilayer , layer_type ,
                               HalfWidH , HalfWidV , padH , padV , strideH , strideV ,
                               layer_weights , height , width , depth , nhid ,
                               this_delta , prior_delta , n_prior_weights ) ;
               else if (layer_type[ilayer+1] == TYPE_POOLAVG  ||  layer_type[ilayer+1] == TYPE_POOLMAX)
                  compute_pooled_delta ( ilayer , layer_type ,
                               PoolWidH , PoolWidV , strideH , strideV ,
                               height , width , depth , nhid ,
                               this_delta , prior_delta , poolmax_id ) ;
               }

            ++k ;
            }  // For width dimension in this hidden layer
         }  // For height dimension in this hidden layer
      }  // For depth dimension in this hidden layer
   
   assert ( k == nhid[ilayer] ) ;
}


/*
--------------------------------------------------------------------------------

   error() - Compute total (not mean) error for a set of training cases

--------------------------------------------------------------------------------
*/

static double batch_error (
   int istart ,                   // Index of first case in batch
   int istop ,                    // And one past last case
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   int layer_type[MAX_LAYERS] ,   // Each entry (input to final) is TYPE_? in CONST.H
   double *output ,               // Put the computed outputs here
   double *predictions ,          // Save predictions here.  Used in CONFUSE.CPP.  Otherwise wasted effort, but not much.
   double *activity[MAX_LAYERS] , // Activity vector for each layer, used only when ilayer>0
   int HalfWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,     // And vertical
   int padH[MAX_LAYERS] ,         // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,         // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   int PoolWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ,     // And vertical
   double *layer_weights[MAX_LAYERS+1] , // Pointers to each layer's weights in 'weight' vector
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   int *poolmax_id[MAX_LAYERS] ,  // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   int n_prior_weights[MAX_LAYERS+1]  // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
)
{
   int i, icase, imax ;
   double err, tot_err, *dptr, tmax ;

   tot_err = 0.0 ;  // Total error will be cumulated here

   for (icase=istart ; icase<istop ; icase++) {  // Do all cases

      dptr = database + icase * n_db_cols ; // Point to this case (database is global, as is n_db_cols)
      trial_thr ( dptr , output , n_layers , layer_type , activity , HalfWidH , HalfWidV ,
                  padH , padV , strideH , strideV , PoolWidH , PoolWidV , poolmax_id ,
                  layer_weights , height , width , depth , nhid , n_prior_weights ) ;
      err = 0.0 ;

      tmax = -1.e30 ;
      imax = 0 ;                       // Not needed; shuts up LINT
      for (i=0 ; i<n_classes ; i++) {  // Find the true class as that having max target
                                       // This is more general than using a single integer class id,
                                       // as it allows for probability-based class membership
         predictions[icase*n_classes+i] = output[i] ;  // Save for other routines (CONFUSE.CPP); usually a waste, though
         if (dptr[n_pred+i] > tmax) {  // n_pred is global
            imax = i ;
            tmax = dptr[n_pred+i] ;
            }
         }
      err = -log ( output[imax] + 1.e-30 ) ;
      tot_err += err ;
      } // for all cases

   return tot_err ;
}


/*
--------------------------------------------------------------------------------

   grad - Compute the total (not mean) gradient for a subset of the training set

   This is called from the wrapper in MODEL.CPP

   In a CONV layer, weight order is:
      Layer depth
         Input slice
            Input height
               Input width

   In a LOCAL layer, weight order is:
      Layer depth
         Layer height
            Layer width
               Input slice
                  Input height
                     Input width

   As each stage of backprop begins, the 'this_delta[j]' array contains the
   derivative of the criterion with respect to the input (post-weight)
   to neuron j in the next layer.

   To get the contribution of that neuron j from neuron i in the current layer,
   the layer whose gradient is currently being computed, we multiply delta[j]
   by the weight connecting current-layer neuron i to next-layer neuron j.
   This gives us the PART of the total derivative due to the output of
   neuron i in the current layer going through neuron j in the next layer.

   But the output of neuron i impacts the criterion derivative through ALL
   neurons in the next layer.  Thus, to get the derivative of the criterion
   with respect to the output of neuron j, we must sum these parts across
   all neurons (values of j) in the next layer.

   To get the derivative of the criterion with respect to the input to neuron i,
   we multiply this sum by the derivative of neuron i's activation function.

   This will be used for propagating delta for the next layer back.

   To get the derivative of the criterion with respect to a weight coming into
   neuron i, we multiply by the input coming through this weight (the output of
   the prior layer's neuron).

--------------------------------------------------------------------------------
*/

static double batch_grad (
   int istart ,                   // Index of first case in batch
   int istop ,                    // And one past last case
   int n_all_weights ,            // Includes bias and final layer weights
   double *gradient ,             // 'n_all_weights' gradient, aligned with weights
   int n_layers ,                 // Number of hidden layers; does not include input or output layer
   int layer_type[MAX_LAYERS] ,   // Each entry (input to final) is TYPE_? in CONST.H
   double *output ,               // Put the computed outputs here
   double *activity[MAX_LAYERS] , // Activity vector for each layer, used only when ilayer>0
   int HalfWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,     // And vertical
   int padH[MAX_LAYERS] ,         // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,         // And vertical
   int strideH[MAX_LAYERS] ,      // Horizontal stride
   int strideV[MAX_LAYERS] ,      // And vertical
   int PoolWidH[MAX_LAYERS] ,     // Horizontal half width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ,     // And vertical
   double *layer_weights[MAX_LAYERS+1] , // Pointers to each layer's weights in 'weight' vector
   double *layer_gradient[MAX_LAYERS+1] , // Pointers to each layer's gradient in 'gradient' vector
   int height[MAX_LAYERS] ,       // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,        // Ditto horizontal
   int depth[MAX_LAYERS] ,        // Number of hidden neurons if fully connected, else number of slices in this layer
   int nhid[MAX_LAYERS] ,         // Total number of neurons in this layer = height times width times depth
   double *this_delta ,           // Scratch vector for gradient computation
   double *prior_delta ,          // Ditto
   int *poolmax_id[MAX_LAYERS] ,  // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   int n_prior_weights[MAX_LAYERS+1]  // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   )
{
   int i, j, icase, ilayer, nprev, nnext, imax ;
   double *dptr, error, *prevact, *gradptr, delta, tmax ;

   for (i=0 ; i<n_all_weights ; i++)  // Zero gradient for summing
      gradient[i] = 0.0 ;             // All layers are strung together here

   error = 0.0 ;  // Will cumulate total error here for return to user

   for (icase=istart ; icase<istop ; icase++) {

      dptr = database + icase * n_db_cols ; // Point to this case (database is global, as is n_db_cols)

/*
   Cumulate error criterion
*/

      trial_thr ( dptr , output , n_layers , layer_type , activity , HalfWidH , HalfWidV ,
                  padH , padV , strideH , strideV , PoolWidH , PoolWidV , poolmax_id ,
                  layer_weights , height , width , depth , nhid , n_prior_weights ) ;

      tmax = -1.e30 ;
      imax = 0 ;                       // Not needed; shuts up LINT
      for (i=0 ; i<n_classes ; i++) {  // Find the true class as that having max target
                                       // This is more general than using a single integer class id,
                                       // as it allows for probability-based class membership
         if (dptr[n_pred+i] > tmax) {  // n_pred is global
            imax = i ;
            tmax = dptr[n_pred+i] ;
            }
         this_delta[i] = dptr[n_pred+i] - output[i] ; // Neg deriv of cross entropy wrt input (logit) i
         }
      error -= log ( output[imax] + 1.e-30 ) ;

/*
   Cumulate output gradient
*/

      if (n_layers == 0) {                  // No hidden layer
         nprev = n_pred ;                   // Number of inputs to the output layer
         prevact = dptr ;                   // Point to this sample
         }
      else {
         nprev = nhid[n_layers-1] ;         // The last hidden layer
         prevact = activity[n_layers-1] ;   // Point to layer feeding the output layer
         }
      assert ( nprev+1 == n_prior_weights[n_layers] ) ;
      gradptr = layer_gradient[n_layers] ;  // Point to output gradient in grand gradient vector
      for (i=0 ; i<n_classes ; i++) {       // For all output neurons
         delta = this_delta[i] ;            // Neg deriv of criterion wrt logit
         for (j=0 ; j<nprev ; j++)
            *gradptr++ += delta * prevact[j] ; // Cumulate for all training cases
         *gradptr++ += delta ;              // Bias activation is always 1
         }

      nnext = n_classes ;                   // Prepare for moving back one layer

/*
   Cumulate hidden gradients.
   Each of these calls also backprops delta from this_delta to prior_delta.
   This is why we also have a call to grad_no_thr_POOL, even though
   a pooled layer has no weights and hence no gradient.
   That call handles backpropping delta just like the other calls.
*/

      for (ilayer=n_layers-1 ; ilayer>=0 ; ilayer--) {   // For each hidden layer, working backwards

         if (layer_type[ilayer] == TYPE_FC)
            grad_thr_FC ( icase , ilayer , n_layers , layer_type ,
                          height , width , depth , HalfWidH , HalfWidV , PoolWidH , PoolWidV ,
                          padH , padV , strideH , strideV , poolmax_id , activity ,
                          layer_weights , layer_gradient , this_delta , prior_delta , nhid , n_prior_weights ) ;

         else if (layer_type[ilayer] == TYPE_LOCAL)
            grad_thr_LOCAL ( icase , ilayer , n_layers , layer_type ,
            height , width , depth , HalfWidH , HalfWidV , PoolWidH , PoolWidV , padH , padV ,
            strideH , strideV , poolmax_id , activity , layer_weights , layer_gradient ,
            this_delta , prior_delta , nhid , n_prior_weights ) ;

         else if (layer_type[ilayer] == TYPE_CONV)
            grad_thr_CONV ( icase , ilayer , n_layers , layer_type ,
            height , width , depth , HalfWidH , HalfWidV , PoolWidH , PoolWidV ,
            padH , padV , strideH , strideV , poolmax_id , activity ,
            layer_weights , layer_gradient , this_delta , prior_delta , nhid , n_prior_weights ) ;

         else if (layer_type[ilayer] == TYPE_POOLAVG  ||  layer_type[ilayer] == TYPE_POOLMAX)
            // POOL has no weights and hence no gradient, but this propagates delta
            grad_thr_POOL ( ilayer , n_layers , layer_type ,
                            height , width , depth , nhid , HalfWidH , HalfWidV ,
                            PoolWidH , PoolWidV , padH , padV , strideH , strideV ,
                            poolmax_id , layer_weights , n_prior_weights , this_delta , prior_delta ) ;


         else
            assert ( 2 == 1 ) ;

         for (i=0 ; i<nhid[ilayer] ; i++)           // These will be delta for the next layer back
            this_delta[i] = prior_delta[i] ;

         }  // For all layers, working backwards

      } // for all cases

   return error ;  // Negative log likelihood
}


/*
--------------------------------------------------------------------------------

   Thread stuff...
      Structure for passing information to/from threaded code
      Threaded code is called by the main subroutine

--------------------------------------------------------------------------------
*/

typedef struct {
   int istart ;             // Index of first case in batch
   int istop ;              // And one past last case
   int n_layers ;           // Number of hidden layers; does not include input or output layer
   int *layer_type ;        // Each entry (input to final) is TYPE_? in CONST.H
   double *output ;         // Put the computed outputs here
   double *predictions ;    // Save predictions here.  Used in CONFUSE.CPP.  Otherwise wasted effort, but not much.
   double **activity ;      // Activity vector for each layer, used only when ilayer>0
   int *HalfWidH ;          // Horizontal half width looking back to prior layer
   int *HalfWidV ;          // And vertical
   int *padH ;              // Horizontal padding, should not exceed half width
   int *padV ;              // And vertical
   int *strideH ;           // Horizontal stride
   int *strideV ;           // And vertical
   int *PoolWidH ;          // Horizontal half width looking back to prior layer
   int *PoolWidV ;          // And vertical
   double **layer_weights ; // Pointers to each layer's weights in 'weight' vector
   int *height ;            // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int *width ;             // Ditto horizontal
   int *depth ;             // Number of hidden neurons if fully connected, else number of slices in this layer
   int *nhid ;              // Total number of neurons in this layer = height times width times depth
   int **poolmax_id ;       // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   int *n_prior_weights ;   // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   double error ;
} ERR_PARAMS ;


static unsigned int __stdcall batch_error_wrapper ( LPVOID dp )
{
((ERR_PARAMS *) dp)->error = batch_error (
   ((ERR_PARAMS *) dp)->istart ,
   ((ERR_PARAMS *) dp)->istop ,
   ((ERR_PARAMS *) dp)->n_layers ,
   ((ERR_PARAMS *) dp)->layer_type ,
   ((ERR_PARAMS *) dp)->output ,
   ((ERR_PARAMS *) dp)->predictions ,
   ((ERR_PARAMS *) dp)->activity ,
   ((ERR_PARAMS *) dp)->HalfWidH ,
   ((ERR_PARAMS *) dp)->HalfWidV ,
   ((ERR_PARAMS *) dp)->padH ,
   ((ERR_PARAMS *) dp)->padV ,
   ((ERR_PARAMS *) dp)->strideH ,
   ((ERR_PARAMS *) dp)->strideV ,
   ((ERR_PARAMS *) dp)->PoolWidH ,
   ((ERR_PARAMS *) dp)->PoolWidV ,
   ((ERR_PARAMS *) dp)->layer_weights ,
   ((ERR_PARAMS *) dp)->height ,
   ((ERR_PARAMS *) dp)->width ,
   ((ERR_PARAMS *) dp)->depth ,
   ((ERR_PARAMS *) dp)->nhid ,
   ((ERR_PARAMS *) dp)->poolmax_id ,
   ((ERR_PARAMS *) dp)->n_prior_weights ) ;
   return 0 ;
}


typedef struct {
   int istart ;              // Index of first case in batch
   int istop ;               // And one past last case
   int n_all_weights ;       // Includes bias and final layer weights
   double *gradient ;        // 'n_all_weights' gradient; aligned with weights
   int n_layers ;            // Number of hidden layers; does not include input or output layer
   int *layer_type ;         // Each entry (input to final) is TYPE_? in CONST.H
   double *output ;          // Put the computed outputs here
   double **activity ;       // Activity vector for each layer; used only when ilayer>0
   int *HalfWidH ;           // Horizontal half width looking back to prior layer
   int *HalfWidV ;           // And vertical
   int *padH ;               // Horizontal padding; should not exceed half width
   int *padV ;               // And vertical
   int *strideH ;            // Horizontal stride
   int *strideV ;            // And vertical
   int *PoolWidH ;           // Horizontal half width looking back to prior layer
   int *PoolWidV ;           // And vertical
   double **layer_weights ;  // Pointers to each layer's weights in 'weight' vector
   double **layer_gradient ; // Pointers to each layer's gradient in 'gradient' vector
   int *height ;             // Number of neurons vertically in a slice of this layer; 1 if fully connected
   int *width ;              // Ditto horizontal
   int *depth ;              // Number of hidden neurons if fully connected; else number of slices in this layer
   int *nhid ;               // Total number of neurons in this layer = height times width times depth
   double *this_delta ;      // Scratch vector for gradient computation
   double *prior_delta ;     // Ditto
   int **poolmax_id ;        // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   int *n_prior_weights ;    // N of inputs per neuron (including bias) to prior layer = (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   double error ;            // Error is returned here
} GRAD_PARAMS ;


static unsigned int __stdcall batch_grad_wrapper ( LPVOID dp )
{
((GRAD_PARAMS *) dp)->error = batch_grad (
   ((GRAD_PARAMS *) dp)->istart ,
   ((GRAD_PARAMS *) dp)->istop ,
   ((GRAD_PARAMS *) dp)->n_all_weights ,
   ((GRAD_PARAMS *) dp)->gradient ,
   ((GRAD_PARAMS *) dp)->n_layers ,
   ((GRAD_PARAMS *) dp)->layer_type ,
   ((GRAD_PARAMS *) dp)->output ,
   ((GRAD_PARAMS *) dp)->activity ,
   ((GRAD_PARAMS *) dp)->HalfWidH ,
   ((GRAD_PARAMS *) dp)->HalfWidV ,
   ((GRAD_PARAMS *) dp)->padH ,
   ((GRAD_PARAMS *) dp)->padV ,
   ((GRAD_PARAMS *) dp)->strideH ,
   ((GRAD_PARAMS *) dp)->strideV ,
   ((GRAD_PARAMS *) dp)->PoolWidH ,
   ((GRAD_PARAMS *) dp)->PoolWidV ,
   ((GRAD_PARAMS *) dp)->layer_weights ,
   ((GRAD_PARAMS *) dp)->layer_gradient ,
   ((GRAD_PARAMS *) dp)->height ,
   ((GRAD_PARAMS *) dp)->width ,
   ((GRAD_PARAMS *) dp)->depth ,
   ((GRAD_PARAMS *) dp)->nhid ,
   ((GRAD_PARAMS *) dp)->this_delta ,
   ((GRAD_PARAMS *) dp)->prior_delta ,
   ((GRAD_PARAMS *) dp)->poolmax_id ,
   ((GRAD_PARAMS *) dp)->n_prior_weights ) ;
   return 0 ;
}


/*
--------------------------------------------------------------------------------

   trial_error() - Compute error for a set of training cases

--------------------------------------------------------------------------------
*/

double Model::trial_error_thr ( int jstart , int jstop )
{
   int i, nc, ret_val, ithread, n_threads, n_in_batch, n_done, istart, istop ;
   int ilayer, ineuron, ivar, n_prior ;
   double error, wpen, *wptr, wt ;
   char msg[256] ;
   ERR_PARAMS params[MAX_THREADS] ;
   HANDLE threads[MAX_THREADS] ;

   nc = jstop - jstart ;

/*
   Initialize parameters that will not change for threads.
*/

   for (i=0 ; i<max_threads ; i++) {
      params[i].n_layers = n_layers ;
      params[i].layer_type = layer_type ;
      params[i].output = thr_output + i * n_classes ;
      params[i].predictions = pred ;
      params[i].activity = thr_activity[i] ;  // See MOD_TRAIN.CPP
      params[i].HalfWidH = HalfWidH ;
      params[i].HalfWidV = HalfWidV ;
      params[i].padH = padH ;
      params[i].padV = padV ;
      params[i].strideH = strideH ;
      params[i].strideV = strideV ;
      params[i].PoolWidH = PoolWidH ;
      params[i].PoolWidV = PoolWidV ;
      params[i].layer_weights = layer_weights ;
      params[i].height = height ;
      params[i].width = width ;
      params[i].depth = depth ;
      params[i].nhid = nhid ;
      params[i].poolmax_id = thr_poolmax_id[i] ;  // See MOD_TRAIN.CPP
      params[i].n_prior_weights = n_prior_weights ;
      }


/*
------------------------------------------------------------------------------------------------

   Batch loop uses a different thread for each batch

------------------------------------------------------------------------------------------------
*/

   n_threads = nc / 100 ;
   if (n_threads < 1)
      n_threads = 1 ;
   if (n_threads > max_threads)
      n_threads = max_threads ;

   istart = jstart ;    // Batch start = training data start
   n_done = 0 ;         // Number of training cases done in this epoch so far

   for (ithread=0 ; ithread<n_threads ; ithread++) {
      n_in_batch = (nc - n_done) / (n_threads - ithread) ;  // Cases left to do / batches left to do
      assert ( n_in_batch > 0 ) ;
      istop = istart + n_in_batch ;                         // Stop just before this index

      // Set the pointers that vary with the batch

      params[ithread].istart = istart ;
      params[ithread].istop = istop ;

      threads[ithread] = (HANDLE) _beginthreadex ( NULL , 0 , batch_error_wrapper , &params[ithread] , 0 , NULL ) ;
      if (threads[ithread] == NULL) {
         audit ( "Internal ERROR: bad thread creation in MOD_THR trial_error_thr()" ) ;
         for (i=0 ; i<n_threads ; i++) {
            if (threads[i] != NULL)
               CloseHandle ( threads[i] ) ;
            }
         return -1.e40 ;
         }

      n_done += n_in_batch ;
      istart = istop ;
      } // For all threads / batches

/*
   Wait for threads to finish
*/

   ret_val = WaitForMultipleObjects ( n_threads , threads , TRUE , 1200000 ) ;
   if (ret_val == WAIT_TIMEOUT  ||  ret_val == WAIT_FAILED  ||  ret_val < 0  ||  ret_val >= n_threads) {
      sprintf_s ( msg, "INTERNAL ERROR!!!  Thread wait 2 failed (%d) in MOD_THR.CPP", ret_val ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      if (ret_val == WAIT_TIMEOUT)
         audit ( "Timeout waiting for computation to finish; problem too large" ) ;
      return -1.e40 ;
      }

   error = 0.0 ;        // Cumulates error
   for (ithread=0 ; ithread<n_threads ; ithread++) {
      error += params[ithread].error ;
      CloseHandle ( threads[ithread] ) ;
      }

/*
   Deal with weight penalty
*/

   wpen = TrainParams.wpen / n_all_weights ;
   penalty = 0.0 ;
   for (ilayer=0 ; ilayer<=n_layers ; ilayer++) {  // Do all hidden layers, plus final
      wptr = layer_weights[ilayer] ;
      n_prior = n_prior_weights[ilayer] ;

      if (ilayer == n_layers) {
         for (ineuron=0 ; ineuron<n_classes ; ineuron++) {
            for (ivar=0 ; ivar<n_prior-1 ; ivar++) {   // Do not include bias in penalty
               wt = wptr[ineuron*n_prior+ivar] ;
               penalty += wt * wt ;
               }
            }
         }

      else if (layer_type[ilayer] == TYPE_FC) {
         for (ineuron=0 ; ineuron<nhid[ilayer] ; ineuron++) {
            for (ivar=0 ; ivar<n_prior-1 ; ivar++) {   // Do not include bias in penalty
               wt = wptr[ineuron*n_prior+ivar] ;
               penalty += wt * wt ;
               }
            }
         }

      else if (layer_type[ilayer] == TYPE_LOCAL) {
         // For LOCAL layers, neuron layout in current layer is (height, width, depth).
         for (ineuron=0 ; ineuron<nhid[ilayer] ; ineuron++) {
            for (ivar=0 ; ivar<n_prior-1 ; ivar++) {   // Do not include bias in penalty
               wt = wptr[ineuron*n_prior+ivar] ;
               penalty += wt * wt ;
               }
            }
         }

      else if (layer_type[ilayer] == TYPE_CONV) {
         // For CONV layers, each depth has its own weight set, but weights across visual field are identical
         for (ineuron=0 ; ineuron<depth[ilayer] ; ineuron++) {
            for (ivar=0 ; ivar<n_prior-1 ; ivar++) {   // Do not include bias in penalty
               wt = wptr[ineuron*n_prior+ivar] ;
               penalty += wt * wt ;
               }
            }
         }
      }

   penalty *= wpen ;
   return error / (nc * n_classes) + penalty ;
}

/*
--------------------------------------------------------------------------------

   grad() - Compute error for a set of training cases

--------------------------------------------------------------------------------
*/

double Model::grad_thr ( int jstart , int jstop )
{
   int i, nc, ret_val, ithread, n_threads, n_in_batch, n_done, istart, istop ;
   int ilayer, ineuron, ivar, n_prior ;
   double error, wpen, wt, *wptr, *gptr ;
   char msg[256] ;
   GRAD_PARAMS params[MAX_THREADS] ;
   HANDLE threads[MAX_THREADS] ;

   nc = jstop - jstart ;

/*
   Initialize parameters that will not change for threads.
*/

   for (i=0 ; i<max_threads ; i++) {
      params[i].n_all_weights = n_all_weights ;
      params[i].gradient = thr_gradient[i] ;  // See MODEL.CPP
      params[i].n_layers = n_layers ;
      params[i].layer_type = layer_type ;
      params[i].output = thr_output + i * n_classes ;
      params[i].activity = thr_activity[i] ;  // See MOD_TRAIN.CPP
      params[i].HalfWidH = HalfWidH ;
      params[i].HalfWidV = HalfWidV ;
      params[i].padH = padH ;
      params[i].padV = padV ;
      params[i].strideH = strideH ;
      params[i].strideV = strideV ;
      params[i].PoolWidH = PoolWidH ;
      params[i].PoolWidV = PoolWidV ;
      params[i].layer_weights = layer_weights ;
      params[i].layer_gradient = thr_layer_gradient[i] ;  // See MOD_TRAIN.CPP
      params[i].height = height ;
      params[i].width = width ;
      params[i].depth = depth ;
      params[i].nhid = nhid ;
      params[i].this_delta = thr_this_delta + i * max_any_layer ;
      params[i].prior_delta = thr_prior_delta + i * max_any_layer ;
      params[i].poolmax_id = thr_poolmax_id[i] ;  // See MOD_TRAIN.CPP
      params[i].n_prior_weights = n_prior_weights ;
      }


/*
------------------------------------------------------------------------------------------------

   Batch loop uses a different thread for each batch

------------------------------------------------------------------------------------------------
*/

   n_threads = nc / 100 ;
   if (n_threads < 1)
      n_threads = 1 ;
   if (n_threads > max_threads)
      n_threads = max_threads ;

   istart = jstart ;    // Batch start = training data start
   n_done = 0 ;         // Number of training cases done in this epoch so far

   for (ithread=0 ; ithread<n_threads ; ithread++) {
      n_in_batch = (nc - n_done) / (n_threads - ithread) ;  // Cases left to do / batches left to do
      assert ( n_in_batch > 0 ) ;
      istop = istart + n_in_batch ;                         // Stop just before this index
      assert ( istop <= jstop ) ;

      // Set the pointers that vary with the batch

      params[ithread].istart = istart ;
      params[ithread].istop = istop ;

      threads[ithread] = (HANDLE) _beginthreadex ( NULL , 0 , batch_grad_wrapper , &params[ithread] , 0 , NULL ) ;
      if (threads[ithread] == NULL) {
         audit ( "Internal ERROR: bad thread creation in MOD_THR grad_thr()" ) ;
         for (i=0 ; i<n_threads ; i++) {
            if (threads[i] != NULL)
               CloseHandle ( threads[i] ) ;
            }
         return -1.e40 ;
         }

      n_done += n_in_batch ;
      istart = istop ;
      } // For all threads / batches

/*
   Wait for threads to finish
*/

   ret_val = WaitForMultipleObjects ( n_threads , threads , TRUE , 1200000 ) ;
   if (ret_val == WAIT_TIMEOUT  ||  ret_val == WAIT_FAILED  ||  ret_val < 0  ||  ret_val >= n_threads) {
      sprintf_s ( msg, "INTERNAL ERROR!!!  Thread wait 2 failed (%d) in MOD_THR.CPP", ret_val ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      if (ret_val == WAIT_TIMEOUT)
         audit ( "Timeout waiting for computation to finish; problem too large" ) ;
      return -1.e40 ;
      }

   error = 0.0 ;        // Cumulates error
   for (i=0 ; i<n_all_weights ; i++)  // Zero gradient for summing
      gradient[i] = 0.0 ;             // All layers are strung together here

   for (ithread=0 ; ithread<n_threads ; ithread++) {
      error += params[ithread].error ;
      for (i=0 ; i<n_all_weights ; i++)
         gradient[i] += params[ithread].gradient[i] ;
      CloseHandle ( threads[ithread] ) ;
      }

   for (i=0 ; i<n_all_weights ; i++)
      gradient[i] /= nc * n_classes ;


/*
   Deal with weight penalty
*/

   wpen = TrainParams.wpen / n_all_weights ;
   penalty = 0.0 ;
   for (ilayer=0 ; ilayer<=n_layers ; ilayer++) {  // Do all hidden layers, plus final
      wptr = layer_weights[ilayer] ;
      gptr = layer_gradient[ilayer] ;
      n_prior = n_prior_weights[ilayer] ;

      if (ilayer == n_layers) {
         for (ineuron=0 ; ineuron<n_classes ; ineuron++) {
            for (ivar=0 ; ivar<n_prior-1 ; ivar++) {   // Do not include bias in penalty
               wt = wptr[ineuron*n_prior+ivar] ;
               penalty += wt * wt ;
               gptr[ineuron*n_prior+ivar] -= 2.0 * wpen * wt ;
               }
            }
         }

      else if (layer_type[ilayer] == TYPE_FC) {
         for (ineuron=0 ; ineuron<nhid[ilayer] ; ineuron++) {
            for (ivar=0 ; ivar<n_prior-1 ; ivar++) {   // Do not include bias in penalty
               wt = wptr[ineuron*n_prior+ivar] ;
               penalty += wt * wt ;
               gptr[ineuron*n_prior+ivar] -= 2.0 * wpen * wt ;
               }
            }
         }

      else if (layer_type[ilayer] == TYPE_LOCAL) {
         // For LOCAL layers, neuron layout in current layer is (height, width, depth).
         for (ineuron=0 ; ineuron<nhid[ilayer] ; ineuron++) {
            for (ivar=0 ; ivar<n_prior-1 ; ivar++) {   // Do not include bias in penalty
               wt = wptr[ineuron*n_prior+ivar] ;
               penalty += wt * wt ;
               gptr[ineuron*n_prior+ivar] -= 2.0 * wpen * wt ;
               }
            }
         }

      else if (layer_type[ilayer] == TYPE_CONV) {
         // For CONV layers, each depth has its own weight set, but weights across visual field are identical
         for (ineuron=0 ; ineuron<depth[ilayer] ; ineuron++) {
            for (ivar=0 ; ivar<n_prior-1 ; ivar++) {   // Do not include bias in penalty
               wt = wptr[ineuron*n_prior+ivar] ;
               penalty += wt * wt ;
               gptr[ineuron*n_prior+ivar] -= 2.0 * wpen * wt ;
               }
            }
         }
      }

   penalty *= wpen ;

   return error / ((jstop - jstart) * n_classes) + penalty ;  // Negative log likelihood
}