/******************************************************************************/
/*                                                                            */
/*  MOD_NO_THR - Assorted model routines computed on host with no threading   */
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

   activity_local - Compute the activation of a LOCAL layer

--------------------------------------------------------------------------------
*/

void Model::activity_local_no_thr ( int ilayer , double *input )
{
   int k, in_row, in_rows, in_col, in_cols, in_slice, in_slices, iheight, iwidth, idepth ;
   int rstart, rstop, cstart, cstop ;
   double sum, *wtptr, *inptr, *outptr, x ;

   assert (ilayer != n_layers) ;     // Output layer is always fully connected

   if (ilayer == 0) {
      in_rows = IMAGE_rows ;
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

void Model::activity_conv_no_thr ( int ilayer , double *input )
{
   int k, in_row, in_rows, in_col, in_cols, in_slice, in_slices, iheight, iwidth, idepth ;
   int rstart, rstop, cstart, cstop ;
   double sum, *wtptr, *inptr, *outptr, x ;

   assert (ilayer != n_layers) ;     // Output layer is always fully connected

   if (ilayer == 0) {
      in_rows = IMAGE_rows ;
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

void Model::activity_fc_no_thr ( int ilayer , double *input , int nonlin )
{
   int iin, iout, nin, nout ;
   double sum, *wtptr, *inptr, *outptr ;

   wtptr = layer_weights[ilayer] ;   // Weights for this layer

   if (ilayer == 0) {
      nin = n_pred ;
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

void Model::activity_pool_no_thr ( int ilayer , double *input )
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
      in_rows = IMAGE_rows ;
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

void Model::trial_no_thr ( double *input )
{
   int i, ilayer ;
   double sum ;

   for (ilayer=0 ; ilayer<n_layers ; ilayer++) {     // These do not include final layer
      if (layer_type[ilayer] == TYPE_LOCAL)
         activity_local_no_thr ( ilayer , input ) ;
      else if (layer_type[ilayer] == TYPE_CONV)
         activity_conv_no_thr ( ilayer , input ) ;
      else if (layer_type[ilayer] == TYPE_FC)
         activity_fc_no_thr ( ilayer , input , 1 ) ;
      else if (layer_type[ilayer] == TYPE_POOLAVG  ||  layer_type[ilayer] == TYPE_POOLMAX)
         activity_pool_no_thr ( ilayer , input ) ;
      else
         assert ( 1 == 2 ) ;
      }
         
   activity_fc_no_thr ( n_layers , input , 0 ) ;

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

   trial_error() - Compute error for a set of training cases

--------------------------------------------------------------------------------
*/

double Model::trial_error_no_thr ( int istart , int istop )
{
   int i, icase, imax, ilayer, ineuron, ivar, n_prior ;
   double err, tot_err, *dptr, tmax, *wptr, wt, wpen ;

   tot_err = 0.0 ;  // Total error will be cumulated here

   for (icase=istart ; icase<istop ; icase++) {  // Do all cases

      dptr = database + icase * n_db_cols ; // Point to this case
      trial_no_thr ( dptr ) ;
      if (icase % 10  &&  (escape_key_pressed || user_pressed_escape()))
         return -1.0 ;   // Flag user escape
      err = 0.0 ;

      tmax = -1.e30 ;
      imax = 0 ;                       // Not needed; shuts up LINT
      for (i=0 ; i<n_classes ; i++) {  // Find the true class as that having max target
                                       // This is more general than using a single integer class id,
                                       // as it allows for probability-based class membership
         pred[icase*n_classes+i] = output[i] ;  // Save for other routines (CONFUSE.CPP, MOD_TRAIN.CPP)
         if (dptr[n_pred+i] > tmax) {
            imax = i ;
            tmax = dptr[n_pred+i] ;
            }
         }
      err = -log ( output[imax] + 1.e-30 ) ;
      tot_err += err ;
      } // for all cases


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
   return tot_err / ((istop - istart) * n_classes) + penalty ;
}

/*
--------------------------------------------------------------------------------

   grad - Compute the gradient for a subset of the training set

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

   As each stage of backprop begins, the 'delta[j]' array contains the
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

   This becomes the delta for the next layer back.

   To get the derivative of the criterion with respect to a weight coming into
   neuron i, we multiply by the input coming through this weight (the output of
   the prior layer's neuron).

--------------------------------------------------------------------------------
*/

double Model::grad_no_thr ( int istart , int istop )
{
   int i, j, icase, ilayer, nprev, nnext, imax, n_prior, ineuron, ivar ;
   double *dptr, error, *prevact, *gradptr, delta, tmax, *wptr, *gptr, wt, wpen ;

   for (i=0 ; i<n_all_weights ; i++)  // Zero gradient for summing
      gradient[i] = 0.0 ;             // All layers are strung together here

   error = 0.0 ;  // Will cumulate total error here for return to user

   for (icase=istart ; icase<istop ; icase++) {

      dptr = database + icase * n_db_cols ; // Point to this case

/*
   Cumulate error criterion
*/

      trial_no_thr ( dptr ) ;
      if (icase % 10  &&  (escape_key_pressed || user_pressed_escape()))
         return -1.0 ;   // Flag user escape

      tmax = -1.e30 ;
      imax = 0 ;                       // Not needed; shuts up LINT
      for (i=0 ; i<n_classes ; i++) {  // Find the true class as that having max target
                                       // This is more general than using a single integer class id,
                                       // as it allows for probability-based class membership
         if (dptr[n_pred+i] > tmax) {
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
            grad_no_thr_FC ( icase , ilayer ) ;

         else if (layer_type[ilayer] == TYPE_LOCAL)
            grad_no_thr_LOCAL ( icase , ilayer ) ;

         else if (layer_type[ilayer] == TYPE_CONV)
            grad_no_thr_CONV ( icase , ilayer ) ;

         else if (layer_type[ilayer] == TYPE_POOLAVG  ||  layer_type[ilayer] == TYPE_POOLMAX)
            grad_no_thr_POOL ( ilayer ) ; // POOL has no weights, hence no gradient, but this propagates delta

         else
            assert ( 2 == 1 ) ;

         for (i=0 ; i<nhid[ilayer] ; i++)           // These will be delta for the next layer back
            this_delta[i] = prior_delta[i] ;

         }  // For all layers, working backwards

      } // for all cases

   for (i=0 ; i<n_all_weights ; i++)
      gradient[i] /= (istop - istart) * n_classes ;


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
   return error / ((istop - istart) * n_classes) + penalty ;  // Negative log likelihood
}


/*
--------------------------------------------------------------------------------

   grad_no_thr_FC - Gradient for a fully connected layer

--------------------------------------------------------------------------------
*/

void Model::grad_no_thr_FC ( int icase , int ilayer )
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
            compute_nonpooled_delta ( ilayer ) ;
         else if (layer_type[ilayer+1] == TYPE_POOLAVG  ||  layer_type[ilayer+1] == TYPE_POOLMAX)
            compute_pooled_delta ( ilayer ) ;
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

void Model::grad_no_thr_LOCAL ( int icase , int ilayer )
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
      in_rows = IMAGE_rows ;     // These, too, are global
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
                  compute_nonpooled_delta ( ilayer ) ;
               else if (layer_type[ilayer+1] == TYPE_POOLAVG  ||  layer_type[ilayer+1] == TYPE_POOLMAX)
                  compute_pooled_delta ( ilayer ) ;
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

void Model::grad_no_thr_CONV ( int icase , int ilayer )
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
                  compute_nonpooled_delta ( ilayer ) ;
               else if (layer_type[ilayer+1] == TYPE_POOLAVG  ||  layer_type[ilayer+1] == TYPE_POOLMAX)
                  compute_pooled_delta ( ilayer ) ;
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

void Model::grad_no_thr_POOL ( int ilayer )
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
                  compute_nonpooled_delta ( ilayer ) ;
               else if (layer_type[ilayer+1] == TYPE_POOLAVG  ||  layer_type[ilayer+1] == TYPE_POOLMAX)
                  compute_pooled_delta ( ilayer ) ;
               }

            ++k ;
            }  // For width dimension in this hidden layer
         }  // For height dimension in this hidden layer
      }  // For depth dimension in this hidden layer
   
   assert ( k == nhid[ilayer] ) ;
}


/*
--------------------------------------------------------------------------------

   compute_nonpooled_delta - Compute all deltas from any layer type to a LOCAL or CONV next layer

--------------------------------------------------------------------------------
*/

void Model::compute_nonpooled_delta ( int ilayer )
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

void Model::compute_pooled_delta ( int ilayer )
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
                     prior_delta[k_this] += this_delta[k_next] * wt ;  // Could pull this multiplication outside
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