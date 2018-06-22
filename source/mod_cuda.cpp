/******************************************************************************/
/*                                                                            */
/*  MOD_CUDA.CPP - host routines for CUDA processing                          */
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

   model_cuda - Compute the mean square error for the entire training set
                and optionally the gradient.

   This is a 'local' worker called by trial_error_cuda() and grad_cuda()
   defined at the end of this module

--------------------------------------------------------------------------------
*/

double Model::model_cuda ( int find_grad , int jstart , int jstop )
{
   int i, nc, ilayer, ret_val, ibatch, n_in_batch, n_subsets, max_batch, istart, istop ;
   int n_done, timer, n_launches, n_prior, ineuron, ivar ;
   double ll, *wptr, *gptr, wt, wpen ;
   char msg[256], error_msg[1024] ;

   nc = jstop - jstart ;

/*
   In order to prevent integer overflow in allocating memory for the gradient
   we compute the minimum number of batches needed to get each batch small enough.
   The CUDA device will allocate max_batch * n_all_weights floats,
   where max_batch is the maximum number of cases in a batch.
   The unit of execution is a single case, so we will compute the gradient
   contribution of each individual case.
   Here, max_batch is the maximum batch size (number of cases in a batch).
*/

   max_batch = MAXPOSNUM / (n_all_weights * sizeof(float)) ;  // Memory allocation size
   if (max_batch > 65535)                               // Grid dimension hardware limitation
      max_batch = 65535 ;

   // The user may want to split into more subsets to prevent CUDA timeout
   if (max_batch > TrainParams.max_batch)
      max_batch = TrainParams.max_batch ;

   n_subsets = (nc + max_batch - 1) / max_batch ;
   assert (n_subsets <= nc) ;

/*
   Initialize CUDA device if not yet done for this session
   We recompute max_batch more accurately, now that we know n_subsets.

   Programming WARNING... If ANY of the parameters in the call to cuda_init change,
                          then cuda_cleanup MUST be called and init redone!
*/

   if (! cuda_initialized) {

      n_done = 0 ;         // Must find max batch size for cuda init
      for (ibatch=0 ; ibatch<n_subsets ; ibatch++) {
         n_in_batch = (nc - n_done) / (n_subsets - ibatch) ;   // Cases left to do / batches left to do
         if (ibatch == 0  ||  n_in_batch > max_batch)
            max_batch = n_in_batch ;
         n_done += n_in_batch ;
         }

      assert (max_batch <= 65535) ;
      assert ( max_batch * sizeof(float) <= MAXPOSNUM / n_all_weights ) ;

      sprintf_s ( msg, "CUDA training will use %d subsets, with max batch size=%d", n_subsets, max_batch ) ;
      audit ( "" ) ;
      audit ( msg ) ;

      ret_val = cuda_init ( n_cases , IMAGE_rows , IMAGE_cols , IMAGE_bands ,
                       n_pred , n_classes , database , max_batch ,
                       TrainParams.max_hid_grad , TrainParams.max_mem_grad ,
                       n_all_weights , n_layers , layer_type , nhid , n_prior_weights ,
                       height , width , depth , HalfWidH , HalfWidV ,
                       padH , padV , strideH , strideV , PoolWidH , PoolWidV ,
                       error_msg ) ;

      if (ret_val == ERROR_INSUFFICIENT_MEMORY) {
         audit ( "" ) ;
         audit ( "ERROR... Host computer has insufficient memory" ) ;
         }
      else if (ret_val == ERROR_CUDA_MEMORY) {
         audit ( "" ) ;
         audit ( "ERROR... CUDA device has insufficient memory" ) ;
         }
      else if (ret_val == ERROR_CUDA_ERROR) {
         audit ( "" ) ;
         audit ( "ERROR... CUDA device had unexpected serious error" ) ;
         }

      if (ret_val) {
         audit ( "" ) ;
         audit ( error_msg ) ;
         audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
         escape_key_pressed = global_abort = 1 ;
         return -1.e40 ;
         }

      cuda_initialized = 1 ;
      } // If not initialized

   if (cuda_weights_changed) {
      ++CudaTimers.ncalls_weights ;
      timer = timeGetTime() ;
      ret_val = cuda_weights_to_device ( n_classes , n_layers , layer_type ,
                                        IMAGE_rows , IMAGE_cols , IMAGE_bands ,
                                        height , width , depth , nhid ,
                                        HalfWidH , HalfWidV , layer_weights ) ;
      CudaTimers.weights += timeGetTime() - timer ;
      cuda_weights_changed = 0 ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR... CUDA device had unexpected serious error" ) ;
         audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
         escape_key_pressed = global_abort = 1 ;
         return -1.e40 ;
         }
      }

/*
--------------------------------------------------------

   Main batch loop

--------------------------------------------------------
*/

   if (find_grad) {
      for (i=0 ; i<n_all_weights ; i++)
         gradient[i] = 0.0 ;
      }

   istart = jstart ;
   n_done = 0 ;         // Number of training cases done in this epoch so far

   for (ibatch=0 ; ibatch<n_subsets ; ibatch++) {
      setpos_progbatch ( (double) ibatch / (double) n_subsets ) ;
      if (escape_key_pressed  ||  user_pressed_escape())
         return -1.0 ;

      n_in_batch = (nc - n_done) / (n_subsets - ibatch) ;   // Cases left to do / batches left to do
      istop = istart + n_in_batch ;                         // Stop just before this index

/*
   Forward pass
*/

      for (ilayer=0 ; ilayer<n_layers ; ilayer++) {         // All hidden layers; we'll do output separately

         timer = timeGetTime() ;
         if (layer_type[ilayer] == TYPE_FC)
            ret_val = cuda_hidden_activation_FC ( istart , istop , nhid[ilayer] , ilayer ) ;
         else if (layer_type[ilayer] == TYPE_LOCAL)
            ret_val = cuda_hidden_activation_LOCAL_CONV_shared ( 1 , istart , istop ,
                      nhid[ilayer] , depth[ilayer] , ilayer ) ;
         else if (layer_type[ilayer] == TYPE_CONV)
            ret_val = cuda_hidden_activation_LOCAL_CONV_shared ( 0 , istart , istop ,
                      nhid[ilayer] , depth[ilayer] , ilayer ) ;
         else if (layer_type[ilayer] == TYPE_POOLAVG)
            ret_val = cuda_hidden_activation_POOLED ( 1 , istart , istop , nhid[ilayer] , depth[ilayer] , ilayer ) ;
         else if (layer_type[ilayer] == TYPE_POOLMAX)
            ret_val = cuda_hidden_activation_POOLED ( 0 , istart , istop , nhid[ilayer] , depth[ilayer] , ilayer ) ;
         CudaTimers.act[ilayer] += timeGetTime() - timer ;
         ++CudaTimers.ncalls_act[ilayer] ;

         if (ret_val) {
            audit ( "" ) ;
            audit ( "ERROR... CUDA device had unexpected serious error" ) ;
            audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
            escape_key_pressed = global_abort = 1 ;
            return -1.e40 ;
            }
         } // For ilayer

/*
   Output layer going forward, then SoftMax
*/

      ++CudaTimers.ncalls_act[n_layers] ;
      timer = timeGetTime() ;
      if (n_layers == 0)
         ret_val = cuda_output_activation_no_hidden ( istart , istop ) ;
      else
         ret_val = cuda_output_activation ( istart , istop ) ;
      CudaTimers.act[n_layers] += timeGetTime() - timer ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR... CUDA device had unexpected serious error" ) ;
         audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
         escape_key_pressed = global_abort = 1 ;
         return -1.e40 ;
         }


      ++CudaTimers.ncalls_softmax ;
      timer = timeGetTime() ;
      ret_val = cuda_softmax ( istart , istop ) ;
      CudaTimers.softmax += timeGetTime() - timer ;
      if (ret_val) {
         audit ( "" ) ;
         audit ( "ERROR... CUDA device had unexpected serious error" ) ;
         audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
         escape_key_pressed = global_abort = 1 ;
         return -1.e40 ;
         }

/*
   Backward pass
*/

      if (find_grad) {

         ret_val = cuda_zero_gradient ( istop-istart , n_all_weights ) ;  // Very fast; no need to time this
         if (ret_val) {
            audit ( "" ) ;
            audit ( "ERROR... CUDA device had unexpected serious error" ) ;
            audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
            escape_key_pressed = global_abort = 1 ;
            return -1.e40 ;
            }

         ++CudaTimers.ncalls_delta[n_layers] ;
         timer = timeGetTime() ;
         ret_val = cuda_output_delta ( istart , istop , n_classes ) ;
         CudaTimers.delta[n_layers] += timeGetTime() - timer ;
         if (ret_val) {
            audit ( "" ) ;
            audit ( "ERROR... CUDA device had unexpected serious error" ) ;
            audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
            escape_key_pressed = global_abort = 1 ;
            return -1.e40 ;
            }

         ++CudaTimers.ncalls_grad[n_layers] ;
         timer = timeGetTime() ;
         if (n_layers == 0)
            ret_val = cuda_output_gradient ( istart , istop-istart , n_pred , -1 , n_classes ) ;
         else
            ret_val = cuda_output_gradient ( istart , istop-istart , nhid[n_layers-1] , n_layers-1 , n_classes ) ;
         CudaTimers.grad[n_layers] += timeGetTime() - timer ;

         if (ret_val) {
            audit ( "" ) ;
            audit ( "ERROR... CUDA device had unexpected serious error" ) ;
            audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
            escape_key_pressed = global_abort = 1 ;
            return -1.e40 ;
            }

         for (ilayer=n_layers-1 ; ilayer>=0 ; ilayer--) {

            // Backprop delta from ilayer+1 to ilayer

            ++CudaTimers.ncalls_delta[ilayer] ;
            timer = timeGetTime() ;
            if (ilayer == n_layers-1  ||  layer_type[ilayer+1] == TYPE_FC)
               ret_val = cuda_backprop_delta_FC ( istop-istart , ilayer , nhid[ilayer] ) ;
            else if (layer_type[ilayer+1] == TYPE_LOCAL  ||  layer_type[ilayer+1] == TYPE_CONV)
               ret_val = cuda_backprop_delta_nonpooled ( istop-istart , ilayer , nhid[ilayer] ) ;
            else if (layer_type[ilayer+1] == TYPE_POOLAVG  ||  layer_type[ilayer+1] == TYPE_POOLMAX)
               ret_val = cuda_backprop_delta_pooled ( istop-istart , ilayer , nhid[ilayer] ) ;
            CudaTimers.delta[ilayer] += timeGetTime() - timer ;
            if (ret_val) {
               audit ( "" ) ;
               audit ( "ERROR... CUDA device had unexpected serious error" ) ;
               audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
               escape_key_pressed = global_abort = 1 ;
               return -1.e40 ;
               }

            // Move delta from prior to this

            ++CudaTimers.ncalls_movedelta ;
            timer = timeGetTime() ;
            ret_val = cuda_move_delta ( istop-istart , nhid[ilayer] ) ;
            CudaTimers.movedelta += timeGetTime() - timer ;
            if (ret_val) {
               audit ( "" ) ;
               audit ( "ERROR... CUDA device had unexpected serious error" ) ;
               audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
               escape_key_pressed = global_abort = 1 ;
               return -1.e40 ;
               }

            // Compute gradient

            timer = timeGetTime() ;
            ret_val = cuda_hidden_gradient ( TrainParams.max_hid_grad , TrainParams.max_mem_grad ,
                                             istart , istop-istart , ilayer ,
                                             layer_type[ilayer] , nhid[ilayer] , ilayer ? nhid[ilayer-1] : n_pred ,
                                             depth[ilayer] , n_prior_weights[ilayer] , &n_launches ) ;
            CudaTimers.grad[ilayer] += timeGetTime() - timer ;
            CudaTimers.ncalls_grad[ilayer] += n_launches ;
            if (ret_val) {
               audit ( "" ) ;
               audit ( "ERROR... CUDA device had unexpected serious error" ) ;
               audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
               escape_key_pressed = global_abort = 1 ;
               return -1.e40 ;
               }
            } // For all layers, going backwards

/*
   Backward pass is complete for this batch
*/

         ++CudaTimers.ncalls_fetchgrad ;
         timer = timeGetTime() ;
         ret_val = cuda_fetch_gradient ( istop-istart , n_all_weights , layer_gradient ,
                                         n_classes , n_layers , layer_type ,
                                         IMAGE_rows , IMAGE_cols , IMAGE_bands ,
                                         height , width , depth , nhid ,
                                         HalfWidH , HalfWidV ) ;
         CudaTimers.fetchgrad += timeGetTime() - timer ;
         if (ret_val) {
            audit ( "" ) ;
            audit ( "ERROR... CUDA device had unexpected serious error" ) ;
            audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
            escape_key_pressed = global_abort = 1 ;
            return -1.e40 ;
            }
         } // If find_grad

      n_done += n_in_batch ;
      istart = istop ;
      } // For ibatch

/*
--------------------------------------------------------

   All batches are processed.

--------------------------------------------------------
*/


   ++CudaTimers.ncalls_ll ;
   timer = timeGetTime() ;
   ret_val = cuda_ll ( nc , &ll ) ;
   CudaTimers.ll += timeGetTime() - timer ;
   if (ret_val) {
      audit ( "" ) ;
      audit ( "ERROR... CUDA device had unexpected serious error" ) ;
      audit ( "ERROR... Unrecoverable serious error... aborting" ) ;
      escape_key_pressed = global_abort = 1 ;
      return -1.e40 ;
      }

   if (find_grad) {
      for (i=0 ; i<n_all_weights ; i++)
         gradient[i] /= (nc * n_classes) ;
      }

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
               if (find_grad)
                  gptr[ineuron*n_prior+ivar] -= 2.0 * wpen * wt ;
               }
            }
         }

      else if (layer_type[ilayer] == TYPE_FC) {
         for (ineuron=0 ; ineuron<nhid[ilayer] ; ineuron++) {
            for (ivar=0 ; ivar<n_prior-1 ; ivar++) {   // Do not include bias in penalty
               wt = wptr[ineuron*n_prior+ivar] ;
               penalty += wt * wt ;
               if (find_grad)
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
               if (find_grad)
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
               if (find_grad)
                  gptr[ineuron*n_prior+ivar] -= 2.0 * wpen * wt ;
               }
            }
         }
      }

   penalty *= wpen ;
   return ll / (nc * n_classes) + penalty ;  // Negative log likelihood
}


double Model::trial_error_cuda ( int jstart , int jstop )
{
   double ret_val ;
   begin_progbatch ( "Trial error" ) ;
   ret_val = model_cuda ( 0 , jstart , jstop ) ;
   end_progbatch () ;
   return ret_val ;
}


double Model::grad_cuda ( int jstart , int jstop )
{
   double ret_val ;
   begin_progbatch ( "Gradient" ) ;
   ret_val = model_cuda ( 1 , jstart , jstop ) ;
   end_progbatch () ;
   return ret_val ;
}
