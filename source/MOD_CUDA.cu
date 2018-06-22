/******************************************************************************/
/*                                                                            */
/*  MOD_CUDA.CU - Core CUDA routines for model training                       */
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

#include <driver_types.h>
#include <cuda_runtime_api.h>

#include "convnet.rh"
#include "const.h"
#include "classes.h"
#include "extern.h"
#include "funcdefs.h"

#define MAX_EXP 300.0

// NOTE... To set up a new project for CUDA, right-click the project name in
//         Solution Explorer, click Build Customizations, select CUDA.
//         Linker needs additional library directory $(CudaToolkitLibDir)
//         Also needs Input/ Additional dependencies cuda.lib;cudart.lib

// This is used as intermediary between device's float and hosts double
static float *fdata = NULL ;

static int n_weights ;            // Total number of weights across all layers
static int n_weights_on_device ;  // Ditto, but extended for 128-byte rows
static int max_convgrad_work ;    // Work area size (# of floats) for CONV gradient, = max_batch * max_convgrad_each
static int max_batch ;            // Max number of cases in a launched batch

// This is strictly for printing memory allocation info for the user

static double total_memory = 0.0 ;



// These are for the reductions used in device_ll
// The number of threads MUST be a power of two!
// The number of blocks given here is a maximum.  The actual number may be less.

#define REDUC_THREADS 256
#define REDUC_BLOCKS 64

// This is for shared memory staging of convolution
#define BLOCK_SIZE 32

static float *reduc_fdata = NULL ;


// These are set in ?_cuda_init and used by the host routine that launches the kernel
// They are basic app parameters, constant for all launches
// Names that begin with d_ are in the device namespace.
// Names that begin with h_ are in the host namespace and equal the device value.
// This lets us save a little time by avoiding the need to pass a bunch of parameters in the launch.
// We could, of course, just pass data pointers as parameters.  But that's overhead.
// So instead we use cudaMemcpyToSymbol() to copy the values in the host namespace
// to values on the device.  This lets __global routines address the values that are
// already set on the device rather than having to use passed parameters.
// The savings is probably small, but worthwhile.

__constant__ int d_ncases ;                // Number of cases in complete training set
__constant__ int d_img_rows ;              // Number of rows in input image
__constant__ int d_img_cols ;              // Number of cols in input image
__constant__ int d_img_bands ;             // Number of bands in input image
__constant__ int d_n_pred ;                // Number of predictors
__constant__ int d_n_classes ;             // Number of classes
__constant__ int d_n_classes_cols ;        // Ditto, extended to multiple of 128 bytes (32 floats) (actual)
__constant__ int d_n_layers ;              // Number of hidden layers; does not include output layer
__constant__ int d_n_weights ;             // Total number of weights across all layers
__constant__ int d_convgrad_cols[MAX_LAYERS] ; // n_prior_weights[ilayer] bumped up to multiple of 32
__constant__ int d_max_convgrad_each ;     // Max hid * convwts_cols in a CONV hid grad launch (work area per case)
                                           // This holds a single case
                                           // See the convgrad_work allocation section for details
                                           // max_convgrad_work = this times max_batch

__constant__ int d_layer_type[MAX_LAYERS] ;// TYPE_? in CONST.H
__constant__ int d_nhid[MAX_LAYERS] ;      // Number of neurons in each of the hidden layers = height*width*depth
__constant__ int d_nhid_cols[MAX_LAYERS] ; // Ditto, extended to multiple of 128 bytes (32 floats) (actual)
__constant__ int d_height[MAX_LAYERS] ;    // Height (rows) of each layer
__constant__ int d_width[MAX_LAYERS] ;     // And width
__constant__ int d_depth[MAX_LAYERS] ;     // And number of slices
__constant__ int d_depth_cols[MAX_LAYERS] ; // Ditto, extended to multiple of 128 bytes (32 floats) (actual); for CONV only
__constant__ int d_n_prior_weights[MAX_LAYERS] ; // N of inputs per neuron (including bias) to prior layer = prior depth * (2*HalfWidH+1) * (2*HalfWidV+1) + 1
                                // A CONV layer has this many weights per layer (slice); a LOCAL layer has this times its nhid

__constant__ int d_HalfWidH[MAX_LAYERS] ;  // Horizontal half width looking back to prior layer
__constant__ int d_HalfWidV[MAX_LAYERS] ;  // And vertical
__constant__ int d_padH[MAX_LAYERS] ;      // Horizontal padding, should not exceed half width
__constant__ int d_padV[MAX_LAYERS] ;      // And vertical
__constant__ int d_strideH[MAX_LAYERS] ;   // Horizontal stride
__constant__ int d_strideV[MAX_LAYERS] ;   // And vertical
__constant__ int d_PoolWidH[MAX_LAYERS] ;  // Horizontal pooling width looking back to prior layer
__constant__ int d_PoolWidV[MAX_LAYERS] ;  // And vertical

static       float *h_predictors = NULL ;  // Raw training data; n_cases by n_pred
__constant__ float *d_predictors ;

static       int *h_class = NULL ;         // Class id is here
__constant__ int *d_class ;

static       double *activations = NULL ;  // Activations of this layer, which we compute
__constant__ double *d_act[MAX_LAYERS] ;   // Pointers to activation vector of each layer

static       double *h_output = NULL ;     // Output activations
__constant__ double *d_output ;

static       int *h_poolmax_id[MAX_LAYERS] ; // Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
__constant__ int *d_poolmax_id[MAX_LAYERS] ; // Pointers to id vector for each layer; NULL for other than POOLMAX layer

static       float *weights = NULL ;       // All weights, including output layer
__constant__ float *d_weights[MAX_LAYERS+1] ; // Pointers to weight vector of each layer, including output

// WARNING... If gradient is ever double instead of float, see MLFN_CUDA.CPP for integer overflow check!
static       float *grad = NULL ;          // Gradient for all weights, including output layer
__constant__ float *d_grad[MAX_LAYERS+1] ; // Pointers to grad vector of each layer, including output
                                           // These are for the first case, and max_batch gradient sets are allocated

static       float *h_convgrad_work = NULL ; // Scratch for unflattened convolution layer gradient
__constant__ float *d_convgrad_work ;

static       double *h_this_delta = NULL ; // Delta for current layer
__constant__ double *d_this_delta ;

static       double *h_prior_delta = NULL ;// Delta for next layer back
__constant__ double *d_prior_delta ;

static       float *h_ll_out = NULL ;
__constant__ float *d_ll_out ;

static cudaDeviceProp deviceProp ;

__global__ void device_hidden_activation_FC ( int istart , int istop , int ilayer ) ;
__global__ void device_hidden_activation_LOCAL_CONV ( int local_vs_conv , int case_start , int case_offset , int slice_start , int n_slices , int ilayer ) ;
__global__ void device_hidden_activation_LOCAL_CONV_shared ( int local_vs_conv , int istart , int ilayer ) ;
__global__ void device_hidden_activation_POOLED ( int avg_vs_max , int istart , int ilayer ) ;
__global__ void device_output_activation_no_hidden ( int istart ) ;
__global__ void device_output_activation ( int istart ) ;
__global__ void device_softmax ( int istart , int istop ) ;
__global__ void device_ll () ;
__global__ void device_output_delta ( int istart ) ;
__global__ void device_output_gradient_no_hidden ( int istart , int nc ) ;
__global__ void device_output_gradient ( int nc , int ilayer ) ;
__global__ void device_backprop_delta_FC ( int ilayer ) ;
__global__ void device_backprop_delta_nonpooled ( int ilayer ) ;
__global__ void device_backprop_delta_pooled ( int ilayer ) ;
__global__ void device_move_delta ( int nhid ) ;
__global__ void device_hidden_gradient_FC ( int istart , int nc , int ilayer ) ;
__global__ void device_hidden_gradient_LOCAL_CONV ( int local_vs_conv , int nfilt , int istart , int depth_offset , int n_depths , int ilayer ) ;
__global__ void device_flatten_gradient ( int islice_start , int max_depth , int ilayer ) ;
__global__ void device_zero_gradient ( int nc ) ;
__global__ void device_fetch_gradient ( int nc ) ;


/*
-----------------------------------------------------

   cuda_init() - Initialize for a model configuration

-----------------------------------------------------
*/

int cuda_init (
   int n_cases ,                // Total number of cases
   int n_img_rows ,             // Number of rows in input image
   int n_img_cols ,             // Number of cols in input image
   int n_img_bands ,            // Number of bands in input image
   int n_pred ,                 // Number of predictors
   int n_classes ,              // Number of classes
   double *data ,               // Ncases by (n_pred+n_classes) data array
   int max_batch_size ,         // Max number of cases that caller wants in a single launch
   int max_hid_grad ,           // Max hid in a CONV hid grad launch; multiple of height*width; <= 65536
   int max_mem_grad ,           // Max memory (bytes) used for CONV scratch storage, which has the potential to be huge
   int n_all_wts ,              // Total number of weights (all layers, including output, and all bias terms)
   int n_layers ,               // Number of layers, not including final
   int layer_type[MAX_LAYERS] , // Each entry (input to final) is TYPE_? in CONST.H
   int nhid[MAX_LAYERS] ,       // Total number of neurons in this layer = height times width times depth
   int n_prior_weights[MAX_LAYERS] , // N of inputs per neuron (including bias) to prior layer = prior depth * (2*HalfWidH+1) * (2*HalfWidV+1) + 1
                                // A CONV layer has this many weights per layer (slice); a LOCAL layer has this times its nhid
   int height[MAX_LAYERS] ,     // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ,      // Ditto horizontal
   int depth[MAX_LAYERS] ,      // Number of hidden neurons if fully connected, else number of slices in this layer
   int HalfWidH[MAX_LAYERS] ,   // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ,   // And vertical
   int padH[MAX_LAYERS] ,       // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ,       // And vertical
   int strideH[MAX_LAYERS] ,    // Horizontal stride
   int strideV[MAX_LAYERS] ,    // And vertical
   int PoolWidH[MAX_LAYERS] ,   // Horizontal pooling width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ,   // And vertical
   char *error_msg              // Returns text of error if problem
   )
{
   int i, j, ilayer, irow, icol, iband, ncols, memsize, n_total, n_max, n_classes_cols ;
   int nhid_cols[MAX_LAYERS], depth_cols[MAX_LAYERS], convgrad_cols[MAX_LAYERS] ;
   int *iclass, ibest, divisor, threads_per_block, batch_size_limit ;
   double best, *xptr, *dptr[MAX_LAYERS+1] ;
   float *fptr[MAX_LAYERS+1] ;
   char msg[256] ;
   cudaError_t error_id ;

   MEMTEXT ( "MOD_CUDA.cu: cuda_init starting" ) ;
   cudalog ( "" ) ;

   max_batch = max_batch_size ;

/*
   Initialize CUDA timers
*/

   for (ilayer=0 ; ilayer<=MAX_LAYERS ; ilayer++) {
      CudaTimers.ncalls_act[ilayer] = 0 ;
      CudaTimers.act[ilayer] = 0 ;
      CudaTimers.ncalls_delta[ilayer] = 0 ;
      CudaTimers.delta[ilayer] = 0 ;
      CudaTimers.ncalls_grad[ilayer] = 0 ;
      CudaTimers.grad[ilayer] = 0 ;
      }

   CudaTimers.ncalls_weights = 0 ;
   CudaTimers.weights = 0 ;
   CudaTimers.ncalls_softmax = 0 ;
   CudaTimers.softmax = 0 ;
   CudaTimers.ncalls_ll = 0 ;
   CudaTimers.ll = 0 ;
   CudaTimers.ncalls_movedelta = 0 ;
   CudaTimers.movedelta = 0 ;
   CudaTimers.ncalls_fetchgrad = 0 ;
   CudaTimers.fetchgrad = 0 ;


   error_id = cudaSetDevice ( cuda_present - 1 ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init SetDevice failed %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      MEMTEXT ( error_msg ) ;
      audit ( error_msg ) ;
      cuda_enable = 0 ;
      return ERROR_CUDA_ERROR ;
      }

   cudaGetDeviceProperties ( &deviceProp , 0 ) ;

/*
   Constants
   We also keep nhid_cols, which is the neurons counts bumped up to multiples of 32 (actual)
   so as to keep rows of weight matrices starting on 128-byte boundaries.
   Ditto for output weights.
   For CONV layers, we bump up depth because every neuron in visual field (height*width) has the same weights in a given slice.
*/

   n_weights = n_all_wts ;
   ncols = n_pred + n_classes ;
   n_classes_cols = (n_classes + 31) / 32 * 32 ; // For memory alignment of weights to 128 bytes
                                                 // This applies to only output weights
   for (i=0 ; i<n_layers ; i++) {
      nhid_cols[i] = (nhid[i] + 31) / 32 * 32 ;
      depth_cols[i] = (depth[i] + 31) / 32 * 32 ;
      h_poolmax_id[i] = NULL ;
      }

   cudaMemcpyToSymbol ( d_ncases , &n_cases , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_img_rows , &n_img_rows , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_img_cols , &n_img_cols , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_img_bands , &n_img_bands , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_n_pred , &n_pred , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_n_classes , &n_classes , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_n_classes_cols , &n_classes_cols , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_n_layers , &n_layers , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_n_weights , &n_weights , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_nhid , nhid , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_nhid_cols , nhid_cols , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_layer_type , layer_type , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_height , height , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_width , width , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_depth , depth , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_depth_cols , depth_cols , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_n_prior_weights , n_prior_weights , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_HalfWidH , HalfWidH , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_HalfWidV , HalfWidV , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_padH , padH , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_padV , padV , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_strideH , strideH , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_strideV , strideV , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_PoolWidH , PoolWidH , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_PoolWidV , PoolWidV , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;


/*
   Set shared memory / cache preferences
*/

   cudaFuncSetCacheConfig ( device_hidden_activation_FC , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_hidden_activation_LOCAL_CONV , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_hidden_activation_LOCAL_CONV_shared , cudaFuncCachePreferNone ) ;
   cudaFuncSetCacheConfig ( device_hidden_activation_POOLED , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_output_activation_no_hidden , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_output_activation , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_softmax , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_ll , cudaFuncCachePreferNone ) ;
   cudaFuncSetCacheConfig ( device_output_delta , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_output_gradient_no_hidden , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_output_gradient , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_backprop_delta_FC , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_backprop_delta_nonpooled , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_backprop_delta_pooled , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_move_delta , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_hidden_gradient_FC , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_hidden_gradient_LOCAL_CONV , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_flatten_gradient , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_zero_gradient , cudaFuncCachePreferL1 ) ;
   cudaFuncSetCacheConfig ( device_fetch_gradient , cudaFuncCachePreferL1 ) ;


/*
   Predictors - We extract only the first n_pred columns from the n_pred+n_classes columns in data
                Reorder them so band changes fastest
*/

   fdata = (float *) MALLOC ( n_cases * n_pred * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   memsize = n_cases * n_pred * sizeof(float) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_predictors , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC predictors = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_predictors, memsize, total_memory / (1024 * 1024) ) ;
   MEMTEXT ( msg ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc predictors (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   j = 0 ;
   for (i=0 ; i<n_cases ; i++) {
      xptr = data + i * ncols ;
      for (irow=0 ; irow<n_img_rows ; irow++) {
         for (icol=0 ; icol<n_img_cols ; icol++) {
            for (iband=0 ; iband<n_img_bands ; iband++)
               fdata[j++] = (float) xptr[(iband*n_img_rows+irow)*n_img_cols+icol] ;
            }
         }
      }
   assert ( j == n_cases * n_pred ) ;

   error_id = cudaMemcpy ( h_predictors , fdata , n_cases * n_pred * sizeof(float) , cudaMemcpyHostToDevice ) ;
   FREE ( fdata ) ;
   fdata = NULL ;

   if (error_id == cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_predictors , &h_predictors , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;
   else {
      sprintf_s ( error_msg , 255 , "CUDA init bad predictors copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Classes; we convert the 1/0 binary output target vector to integer classes
*/

   iclass = (int *) MALLOC ( n_cases * sizeof(int) ) ;
   if (iclass == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   memsize = n_cases * sizeof(int) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_class , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC class = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_class, memsize, total_memory / (1024 * 1024) ) ;
   MEMTEXT ( msg ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc class (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   for (i=0 ; i<n_cases ; i++) {
      best = -1.e60 ;
      xptr = data + i * ncols + n_pred ;
      for (j=0 ; j<n_classes ; j++) {
         if (xptr[j] > best) {
            best = xptr[j] ;
            ibest = j ;
            }
         }
      iclass[i] = ibest ;
      }
         
   error_id = cudaMemcpy ( h_class , iclass , n_cases * sizeof(int) , cudaMemcpyHostToDevice ) ;

   if (error_id == cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_class , &h_class , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;
   else {
      sprintf_s ( error_msg , 255 , "CUDA init bad class copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }

   FREE ( iclass ) ;


/*
   Activations (hidden layers only) ordered (row, col, slice)
*/

   if (n_layers) {
      n_total = 0 ;
      for (i=0 ; i<n_layers ; i++)  // All hidden layers, but not output
         n_total += nhid[i] ;

      memsize = n_total * max_batch * sizeof(double) ;
      total_memory += memsize ;
      error_id = cudaMalloc ( (void **) &activations , (size_t) memsize ) ;
      sprintf_s ( msg, 255 , "CUDA MALLOC activations = %llx  (%d bytes, total=%.2lf MB)",
                  (unsigned long long) activations, memsize, total_memory / (1024 * 1024) ) ;
      MEMTEXT ( msg ) ;
      cudalog ( msg ) ;
      if (error_id  !=  cudaSuccess) {
         sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc activations (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
         return ERROR_CUDA_MEMORY ;
         }

      n_total = 0 ;
      for (i=0 ; i<n_layers ; i++) {
         dptr[i] = activations + n_total * max_batch ;
         n_total += nhid[i] ;
         }

      error_id = cudaMemcpyToSymbol ( d_act , &dptr[0] , n_layers * sizeof(double *) , 0 , cudaMemcpyHostToDevice ) ;
      if (error_id  !=  cudaSuccess) {
         sprintf_s ( error_msg , 255 , "CUDA init bad act ptr copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         return ERROR_CUDA_ERROR ;
         }
      }

   else
      activations = NULL ;


/*
   poolmax_id (POOLMAX layers only) ordered (row, col, slice)
*/

   for (ilayer=0 ; ilayer<n_layers ; ilayer++) {
      if (layer_type[ilayer] == TYPE_POOLMAX) {
         memsize = nhid[ilayer] * max_batch * sizeof(int) ;
         total_memory += memsize ;
         error_id = cudaMalloc ( (void **) &h_poolmax_id[ilayer] , (size_t) memsize ) ;
         sprintf_s ( msg, 255 , "CUDA MALLOC Layer %d poolmax_id = %llx  (%d bytes, total=%.2lf MB)",
                     ilayer, (unsigned long long) h_poolmax_id[ilayer], memsize, total_memory / (1024 * 1024) ) ;
         MEMTEXT ( msg ) ;
         cudalog ( msg ) ;
         if (error_id  !=  cudaSuccess) {
            sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc poolmax_id (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
            return ERROR_CUDA_MEMORY ;
            }
         }
      else
         h_poolmax_id[ilayer] = NULL ;
      }

   error_id = cudaMemcpyToSymbol ( d_poolmax_id , &h_poolmax_id[0] , n_layers * sizeof(int *) , 0 , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad poolmax_id ptr copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Outputs
*/

   memsize = n_cases * n_classes * sizeof(double) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_output , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC output = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_output, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_output , &h_output , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;
   else {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc output (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }


/*
   Weights

   These are stored as the transpose of those in Host,
   with the neurons in the 'current' layer changing fastest.
   Within each layer's weight matrix, rows (sets of current layer weights)
   are stored starting on 128-byte boundaries.
   Thus, n_weights_on_device is generally larger than n_weights, because it
   takes into account row padding.
   Neuron layout in each layer is (height, width, depth).
*/

   n_weights_on_device = 0 ;
   for (ilayer=0 ; ilayer<= n_layers ; ilayer++) { // For each of the hidden layers, plus the final
      if (ilayer == n_layers)
         n_weights_on_device += n_classes_cols * n_prior_weights[ilayer] ;
      else if (layer_type[ilayer] == TYPE_FC  ||  layer_type[ilayer] == TYPE_LOCAL)
         n_weights_on_device += nhid_cols[ilayer] * n_prior_weights[ilayer] ;  // Add in weights for this layer
      else if (layer_type[ilayer] == TYPE_CONV)
         n_weights_on_device += depth_cols[ilayer] * n_prior_weights[ilayer] ; // A convolution layer uses the same weights for every hidden neuron in visible field
      else if (layer_type[i] == TYPE_POOLAVG  ||  layer_type[i] == TYPE_POOLMAX)
         n_weights_on_device += 0 ;                                       // Just for clarity; pooling has no trainable weights
      } // For ilayer

   memsize = n_weights_on_device * sizeof(float) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &weights , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC weights = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) weights, memsize, total_memory / (1024 * 1024) ) ;
   MEMTEXT ( msg ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc weights (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   n_total = 0 ;
   for (ilayer=0 ; ; ilayer++) {            // For each of the hidden layers, plus the final
      fptr[ilayer] = weights + n_total ;
      if (ilayer >= n_layers)
         break ;
      if (layer_type[ilayer] == TYPE_FC  ||  layer_type[ilayer] == TYPE_LOCAL)
         n_total += nhid_cols[ilayer] * n_prior_weights[ilayer] ;  // Add in weights for this layer
      else if (layer_type[ilayer] == TYPE_CONV)
         n_total += depth_cols[ilayer] * n_prior_weights[ilayer] ; // A convolution layer uses the same weights for every hidden neuron in visible field in a slice
      else if (layer_type[i] == TYPE_POOLAVG  ||  layer_type[i] == TYPE_POOLMAX)
         n_total += 0 ;                                       // Just for clarity; pooling has no trainable weights
      } // For ilayer

   error_id = cudaMemcpyToSymbol ( d_weights , &fptr[0] , (n_layers+1) * sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad weight ptr copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   Gradient

   We allocate for max_batch complete gradient vectors, and d_grad will be pointers to the first set.
   Subsequent sets are addressed by adding k * n_weights to the first set.
*/

   memsize = n_weights * max_batch * sizeof(float) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &grad , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC grad = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) grad, memsize, total_memory / (1024 * 1024) ) ;
   MEMTEXT ( msg ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc grad (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   n_total = 0 ;
   for (ilayer=0 ; ; ilayer++) {            // For each of the hidden layers, plus the final
      fptr[ilayer] = grad + n_total ;
      if (ilayer >= n_layers)
         break ;
      if (layer_type[ilayer] == TYPE_FC  ||  layer_type[ilayer] == TYPE_LOCAL)
         n_total += nhid[ilayer] * n_prior_weights[ilayer] ;  // Add in grad for this layer
      else if (layer_type[ilayer] == TYPE_CONV)
         n_total += depth[ilayer] * n_prior_weights[ilayer] ; // A convolution layer uses the same grad for every hidden neuron in visible field
      else if (layer_type[i] == TYPE_POOLAVG  ||  layer_type[i] == TYPE_POOLMAX)
         n_total += 0 ;                                       // Just for clarity; pooling has no trainable grad
      } // For ilayer (each hidden layer)

   error_id = cudaMemcpyToSymbol ( d_grad , &fptr[0] , (n_layers+1) * sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad weight ptr copy %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_ERROR ;
      }


/*
   convgrad_work - Scratch vector for unflattened convolution layer gradient
*/

   max_convgrad_work = 0 ;  // Will find the max work area needed
   for (ilayer=0 ; ilayer<n_layers ; ilayer++) {
      if (layer_type[ilayer] == TYPE_CONV) {
         convgrad_cols[ilayer] = (n_prior_weights[ilayer] + 31) / 32 * 32 ; // CONV scratch is zero padded for full coalescing
         n_max = 1024 * 1024 * max_mem_grad / (max_batch * convgrad_cols[ilayer] * sizeof(float)) ; // Launch limit satisfying memory
         divisor = 1 ; // Figure out how much we have to divide slices to meet max_hid_grad and max_mem_grad limits
         for ( ;; ) {
            j = depth[ilayer] / divisor * height[ilayer] * width[ilayer] ; // We will launch this many hid at a time
            if (j <= max_hid_grad  &&  j <= n_max)
               break ;
            ++divisor ;
            }
         j = depth[ilayer] / divisor * height[ilayer] * width[ilayer] ; // We will launch this many hid at a time
         if (j < height[ilayer] * width[ilayer])   // Careless user specified it too small, so ignore request
            j = height[ilayer] * width[ilayer] ;
         // At this time, j is the number of hidden neurons per launch
         if (j * convgrad_cols[ilayer] > max_convgrad_work)
            max_convgrad_work = j * convgrad_cols[ilayer] ;  // This many weights will be computed in a launch (per case)

         // Print info for user
         cudalog ( "" ) ;
         sprintf_s ( msg, "Gradient computation for layer %d will use %d launches, each max %d hidden neurons",
                     ilayer+1, (depth[ilayer] * height[ilayer] * width[ilayer] + j - 1) / j, j ) ;
         cudalog ( msg ) ;

         threads_per_block = (n_prior_weights[ilayer] + 31) / 32 * 32 ;
         if (threads_per_block > 4 * 32)
            threads_per_block = 4 * 32 ;
         sprintf_s ( msg,  "Launch parameters: Threads per block=%d with %d thread (x) blocks",
                     threads_per_block, (n_prior_weights[ilayer] + threads_per_block - 1) / threads_per_block) ;
         cudalog ( msg ) ;
         sprintf_s ( msg,  "  Max Y dimension (n hidden) = %d; max Z dimension (cases) = %d", j, max_batch_size ) ;
         cudalog ( msg ) ;
         }
      else
         convgrad_cols[ilayer] = 0 ;  // Not needed
      }

   cudaMemcpyToSymbol ( d_max_convgrad_each , &max_convgrad_work , sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;
   cudaMemcpyToSymbol ( d_convgrad_cols , convgrad_cols , n_layers * sizeof(int) , 0 , cudaMemcpyHostToDevice ) ;

   // For storing gradient, we need prior weights and cases in batch

   if (max_convgrad_work) {
      // Must not have integer overflow in memory size
      // At this moment, max_convgrad_work is the max number of weights (neurons times prior) in a launch
      batch_size_limit = MAXPOSNUM / (max_convgrad_work * sizeof(float)) ;  // Memory allocation size
      if (max_batch > batch_size_limit) {
         audit ( "ERROR... User specified number of training cases per subset too large.  Please reduce." ) ;
         cudalog ( "Device initialization error: training cases per subset too large." ) ;
         sprintf_s ( error_msg , 255 , "User ERROR: Architecture and CUDA params limit subset to %d cases", batch_size_limit ) ;
         return ERROR_CUDA_MEMORY ;
         }
      max_convgrad_work *= max_batch ;
      memsize = max_convgrad_work * sizeof(float) ;
      total_memory += memsize ;
      error_id = cudaMalloc ( (void **) &h_convgrad_work , (size_t) memsize ) ;
      sprintf_s ( msg, 255 , "CUDA MALLOC convgrad_work = %llx  (%d bytes, total=%.2lf MB)",
                  (unsigned long long) h_convgrad_work, memsize, total_memory / (1024 * 1024) ) ;
      cudalog ( msg ) ;
      if (error_id  ==  cudaSuccess)
         error_id = cudaMemcpyToSymbol ( d_convgrad_work , &h_convgrad_work , sizeof(float *) , 0 , cudaMemcpyHostToDevice ) ;
      else {
         sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc convgrad_work (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
         return ERROR_CUDA_MEMORY ;
         }
      }
   else
      h_convgrad_work = NULL ;


/*
   This delta, next delta
*/

   n_max = n_classes ;
   for (i=0 ; i<n_layers ; i++) {
      if (nhid[i] > n_max)
         n_max = nhid[i] ;
      }

   memsize = n_max * max_batch * sizeof(double) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_this_delta , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC this_delta = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_this_delta, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_this_delta , &h_this_delta , sizeof(double *) , 0 , cudaMemcpyHostToDevice ) ;
   else {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc this_delta (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }

   memsize = n_max * max_batch * sizeof(double) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_prior_delta , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC prior_delta = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_prior_delta, memsize, total_memory / (1024 * 1024) ) ;
   cudalog ( msg ) ;
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpyToSymbol ( d_prior_delta , &h_prior_delta , sizeof(double *) , 0 , cudaMemcpyHostToDevice ) ;
   else {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc prior_delta (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }


/*
   Log likelihood reduction stuff
*/

   memsize = REDUC_BLOCKS * sizeof(float) ;
   total_memory += memsize ;
   error_id = cudaMalloc ( (void **) &h_ll_out , (size_t) memsize ) ;
   sprintf_s ( msg, 255 , "CUDA MALLOC ll_out = %llx  (%d bytes, total=%.2lf MB)",
               (unsigned long long) h_ll_out, memsize, total_memory / (1024 * 1024) ) ;
   MEMTEXT ( msg ) ;
   cudalog ( msg ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( error_msg , 255 , "CUDA init bad cudaMalloc ll_out (%d): %s", error_id, cudaGetErrorString(error_id) ) ;
      return ERROR_CUDA_MEMORY ;
      }
   cudaMemcpyToSymbol ( d_ll_out , &h_ll_out , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) ;

   MEMTEXT ( "CUDA init reduc_fdata" ) ;
   reduc_fdata = (float *) MALLOC ( REDUC_BLOCKS * sizeof(float) ) ;
   if (reduc_fdata == NULL) {
      sprintf_s ( error_msg , 255 , "CUDA init bad MALLOC reduc_fdata" ) ;
      return ERROR_INSUFFICIENT_MEMORY ;  // New error return
      }



/*
   Allocate fdata large enough to handle all subsequent double <-> float transactions
   This remains allocated until cuda_cleanup() is called, because it is used often in launches.
*/

   n_max = max_convgrad_work ;
   if (n_weights_on_device > n_max)
      n_max = n_weights_on_device ;

   fdata = (float *) MALLOC ( n_max * sizeof(float) ) ;
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_MEMORY ;

   MEMTEXT ( "MOD_CUDA.cu: cuda_init ending" ) ;

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   cuda_weights_to_device - Called from MOD_CUDA.CPP to copy weights

   HOST weights:

   In a CONV layer, weight order is:
      Layer depth
         Input slice
            Input height
               Input width
         Bias

   In a LOCAL layer, weight order is:
      Layer depth
         Layer height
            Layer width
               Input slice
                  Input height
                     Input width
               Bias

   CUDA weights:

   In a CONV layer, weight order is:
      Input height
         Input width
            Input slice
      Bias
               Layer depth
               Pad so layer depth is a multiple of 128

   In a LOCAL layer, weight order is:
      Input height
         Input width
            Input slice
      Bias
               Layer height
                  Layer width
                     Layer depth
               Pad so nhid = layer height*width*depth is a multiple of 128

    A fully connected layer has height=width=1; all neurons are depth.

--------------------------------------------------------------------------------
*/

int cuda_weights_to_device (
   int n_classes ,     // Number of outputs
   int n_layers ,      // Hidden layers; does not include output
   int *layer_type ,   // Each entry (input to final) is TYPE_? in CONST.H
   int img_rows ,      // Size of input image
   int img_cols ,
   int img_bands ,
   int *height ,       // Height of visible field in each layer
   int *width ,        // Width of visible field
   int *depth ,        // Number of slices in each layer
   int *nhid ,         // Number of hidden neurons in each layer
   int *hwH ,          // Half-width of filters
   int *hwV ,
   double **host_weights )  // Vector of pointers to weights for each layer
{
   int n, n_prior, ilayer, ineuron, isub, n_cols_each ;
   int idepth, iheight, iwidth, ndepth, nheight, nwidth ;
   int in_row, in_col, in_slice, in_n_height, in_n_width, in_n_depth ;
   double *wptr ;
   float *fptr ;
   char msg[256] ;
   cudaError_t error_id ;
   
   fptr = fdata ;

   for (ilayer=0 ; ilayer<=n_layers ; ilayer++) {
      wptr = host_weights[ilayer] ;

/*
   Fully connected
*/

      if (ilayer == n_layers  ||  layer_type[ilayer] == TYPE_FC) {
         if (ilayer == 0) {
            in_n_height = img_rows ;
            in_n_width = img_cols ;
            in_n_depth = img_bands ;
            }
         else {
            in_n_height = height[ilayer-1] ;
            in_n_width = width[ilayer-1] ;
            in_n_depth = depth[ilayer-1] ;
            }
         n_prior = in_n_height * in_n_width * in_n_depth + 1 ;  // Number of weights per neuron, including bias
         if (ilayer == n_layers)
            n = n_classes ;  // Equals depth
         else
            n = nhid[ilayer] ;  // Equals depth
         n_cols_each = (n + 31) / 32 * 32 ;  // For memory alignment to 128 bytes
         for (in_row=0 ; in_row<in_n_height ; in_row++) {
            for (in_col=0 ; in_col<in_n_width ; in_col++) {
               for (in_slice=0 ; in_slice<in_n_depth ; in_slice++) {
                  for (idepth=0 ; idepth<n ; idepth++) {
                     // Compute location of this neuron's weight vector in host
                     isub = idepth * n_prior + (in_slice * in_n_height + in_row) * in_n_width + in_col ;
                     *fptr++ = (float) wptr[isub] ;
                     } // For idepth
                  while (idepth++ < n_cols_each)  // Pad to multiple of 128 bytes
                     *fptr++ = 0.0f ;
                  } // For in_slice
               } // For in_col
            } // For in_row

         // Bias
         for (idepth=0 ; idepth<n ; idepth++) {
            // Compute location of this neuron's bias in host
            isub = idepth * n_prior + n_prior - 1 ;
            *fptr++ = (float) wptr[isub] ;
            } // For idepth
         while (idepth++ < n_cols_each)  // Pad to multiple of 128 bytes
            *fptr++ = 0.0f ;
         }

/*
   LOCAL
*/

      else if (layer_type[ilayer] == TYPE_LOCAL) {
         // For LOCAL layers, neuron layout in current layer is (height, width, depth).
         n = nhid[ilayer] ;
         n_cols_each = (n + 31) / 32 * 32 ;  // For memory alignment to 128 bytes
         ndepth = depth[ilayer] ;
         nheight = height[ilayer] ;
         nwidth = width[ilayer] ;
         in_n_height = 2 * hwV[ilayer] + 1 ;
         in_n_width = 2 * hwH[ilayer] + 1 ;
         if (ilayer == 0)
            in_n_depth = img_bands ;
         else
            in_n_depth = depth[ilayer-1] ;
         n_prior = in_n_height * in_n_width * in_n_depth + 1 ;  // Number of weights per neuron, including bias
         for (in_row=0 ; in_row<in_n_height ; in_row++) {
            for (in_col=0 ; in_col<in_n_width ; in_col++) {
               for (in_slice=0 ; in_slice<in_n_depth ; in_slice++) {
                  for (iheight=0 ; iheight<nheight ; iheight++) {  // nhid = ndepth * nheight * nwidth
                     for (iwidth=0 ; iwidth<nwidth ; iwidth++) {   // We must reorder so depth changes fastest
                        for (idepth=0 ; idepth<ndepth ; idepth++) {
                           // Compute location of this neuron's weight in host
                           isub = (idepth * nheight + iheight) * nwidth + iwidth ; // Neuron in this layer
                           isub = isub * n_prior + (in_slice * in_n_height + in_row) * in_n_width + in_col ;
                           *fptr++ = (float) wptr[isub] ;
                           } // For idepth
                        } // For iwidth
                     } // For iheight
                  ineuron = nhid[ilayer] ;
                  while (ineuron++ < n_cols_each)  // Pad to multiple of 128 bytes
                     *fptr++ = 0.0f ;
                  } // For in_slice
               } // For in_col
            } // For in_row

         // Bias
         for (iheight=0 ; iheight<nheight ; iheight++) {  // nhid = ndepth * nheight * nwidth
            for (iwidth=0 ; iwidth<nwidth ; iwidth++) {   // We must reorder so depth changes fastest
               for (idepth=0 ; idepth<ndepth ; idepth++) {
                  // Compute location of this neuron's weight vector in host
                  isub = (idepth * nheight + iheight) * nwidth + iwidth ; // Neuron in this layer
                  isub = isub * n_prior + n_prior - 1 ;
                  *fptr++ = (float) wptr[isub] ;
                  } // For idepth
               } // For iwidth
            } // For iheight
         ineuron = nhid[ilayer] ;
         while (ineuron++ < n_cols_each)  // Pad to multiple of 128 bytes
            *fptr++ = 0.0f ;
         }

/*
   CONV
*/

      else if (layer_type[ilayer] == TYPE_CONV) {
         nheight = height[ilayer] ;
         nwidth = width[ilayer] ;
         ndepth = depth[ilayer] ;
         n_cols_each = (ndepth + 31) / 32 * 32 ;  // For memory alignment to 128 bytes
         in_n_height = 2 * hwV[ilayer] + 1 ;
         in_n_width = 2 * hwH[ilayer] + 1 ;
         if (ilayer == 0)
            in_n_depth = img_bands ;
         else
            in_n_depth = depth[ilayer-1] ;
         n_prior = in_n_height * in_n_width * in_n_depth + 1 ;  // Number of weights per neuron, including bias
         for (in_row=0 ; in_row<in_n_height ; in_row++) {
            for (in_col=0 ; in_col<in_n_width ; in_col++) {
               for (in_slice=0 ; in_slice<in_n_depth ; in_slice++) {
                  for (idepth=0 ; idepth<ndepth ; idepth++) {
                     // Compute location of this neuron's weight vector in host
                     isub = idepth * n_prior + (in_slice * in_n_height + in_row) * in_n_width + in_col ;
                     *fptr++ = (float) wptr[isub] ;
                     } // For idepth
                  while (idepth++ < n_cols_each)  // Pad to multiple of 128 bytes
                     *fptr++ = 0.0f ;
                  } // For in_slice
               } // For in_col
            } // For in_row

         //Bias
         for (idepth=0 ; idepth<ndepth ; idepth++) {
            // Compute location of this neuron's bias in host
            isub = idepth * n_prior + n_prior - 1 ;
            *fptr++ = (float) wptr[isub] ;
            } // For idepth
         while (idepth++ < n_cols_each)  // Pad to multiple of 128 bytes
            *fptr++ = 0.0f ;
         }

      else if (layer_type[ilayer] == TYPE_POOLAVG  ||  layer_type[ilayer] == TYPE_POOLMAX) {
         n = 0 ;  // Not needed.  Just for clarity.
         }


      } // For ilayer

   assert ( fptr == fdata + n_weights_on_device ) ;

   error_id = cudaMemcpy ( weights , fdata , n_weights_on_device * sizeof(float) , cudaMemcpyHostToDevice ) ;
   if (error_id  !=  cudaSuccess) {
      sprintf_s ( msg , 255 , "CUDA ERROR: bad weights_to_device hid %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( "" ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return ERROR_CUDA_ERROR ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   hidden_activation_FC - Compute activations for an FC hidden layer

--------------------------------------------------------------------------------
*/

__global__ void device_hidden_activation_FC (
   int istart ,       // First case in this batch
   int istop ,        // One past last case
   int ilayer         // Layer to process
   )
{
   int icase, ihid, i_input, n_inputs, nhid_cols ;
   float *f_inptr, *wptr ;
   double sum, *actptr, *d_inptr ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;

   if (ihid >= d_nhid[ilayer])
      return ;

   nhid_cols = d_nhid_cols[ilayer] ;

   icase = blockIdx.y ;

   wptr = d_weights[ilayer] + ihid ;  // Device weights are transpose of host weights, with this neuron changing fastest
   sum = 0.0 ;

   if (ilayer == 0) {
      n_inputs = d_n_pred ;
      f_inptr = d_predictors + (icase+istart)*n_inputs ;
      for (i_input=0 ; i_input<n_inputs ; i_input++) {
         sum += *wptr * f_inptr[i_input] ;
         wptr += nhid_cols ;
         }
      sum += *wptr ;   // Bias
      }

   else {
      n_inputs = d_nhid[ilayer-1] ;
      d_inptr = d_act[ilayer-1] + icase*n_inputs ;
      for (i_input=0 ; i_input<n_inputs ; i_input++) {
         sum += *wptr * d_inptr[i_input] ;
         wptr += nhid_cols ;
         }
      sum += *wptr ;   // Bias
      }

   if (sum > MAX_EXP)
      sum = 1.0 ;
   else {
      sum = exp ( 2.0 * sum ) ;
      sum = (sum - 1.0) / (sum + 1.0) ;
      }
   actptr = d_act[ilayer] ;
   actptr[icase*d_nhid[ilayer]+ihid] = sum ;
}

int cuda_hidden_activation_FC (
   int istart ,    // First case in this batch
   int istop ,     // One past last case
   int nhid ,      // Number of hidden neurons in this layer
   int ilayer      // Layer to process
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nhid + threads_per_block - 1) / threads_per_block ;
   block_launch.y = istop - istart ;
   block_launch.z = 1 ;

   device_hidden_activation_FC <<< block_launch , threads_per_block >>> ( istart , istop , ilayer ) ;   

   // This does not trigger an escape, but it keeps the message queue running
   user_pressed_escape () ;

   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_hidden_activation_FC launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
---------------------------------------------------------------------------------

   hidden_activation_LOCAL_CONV - Activations for a LOCAL or CONV hidden layer

---------------------------------------------------------------------------------
*/

__global__ void device_hidden_activation_LOCAL_CONV (
   int local_vs_conv , // Is this a LOCAL (vs CONV) layer?
   int case_start ,    // First case in this batch (relative to dataset)
   int case_offset ,   // Offset relative to this batch (used in shared version)
   int slice_start ,   // First slice in this batch
   int n_slices ,      // Number of slices to be done in this launch
   int ilayer          // Layer to process
   )
{
   int kwt, kin, wtsub, insub, iheight, iwidth, idepth, n_height, n_width, n_depth, wt_cols, ihid ;
   int rstart, rstop, cstart, cstop, rbase, cbase, in_slice, in_row, in_col, nH ;
   float *f_inptr, *wptr ;
   double sum, *actptr ;

   idepth = blockIdx.x * blockDim.x + threadIdx.x ;

   if (idepth >= n_slices)
      return ;

   idepth += slice_start ;
   iheight = blockIdx.y / d_width[ilayer] ;
   iwidth = blockIdx.y % d_width[ilayer] ;

   nH = 2 * d_HalfWidH[ilayer] + 1 ;

   // We are about to compute the activation of neuron (iheight, iwidth, idepth) in this layer.
   // Note that it is critical that idepth be associated with the thread.
   // This ensures that adjacent threads reference the same input, which allows efficient memory use.
   // Also, the weights are ordered so that depth-fastest changes produce perfect or very good coalescing.
   // Thus, neuron layout in current layer is (height, width, depth).
   // This gives strong motivation for LOCAL layers to have depth a multiple of 32.
   // To see why, note the ihid= below.  That multiplication ensures perfect coalescing of the weight fetches.

   // icase = blockIdx.z ;   // Avoid having to declare this (and use a register) by directly referencing it later

   if (local_vs_conv) {
      wt_cols = d_nhid_cols[ilayer] ; // Padded size of weight matrix rows; each has nhid data values, then zero padding
                                      // There are n_prior_weights rows (prior depth * (2*HalfWidH+1) * (2*HalfWidV+1) + 1)
      ihid = (iheight * d_width[ilayer] + iwidth) * d_depth[ilayer] + idepth ;
      wptr = d_weights[ilayer] + ihid ;  // Device weights are transpose of host weights, with this neuron changing fastest
      }                                  // Order is (height, width, depth)

   else {
      wt_cols = d_depth_cols[ilayer] ; // Padded size of weight matrix rows; each has depth[ilayer] data values, then zero padding
                                       // There are n_prior_weights rows (prior depth * (2*HalfWidH+1) * (2*HalfWidV+1) + 1)
                                       // A convolutional layer has a different weight set for each slice,
                                       // but the same weight set for all neurons (visual field placement) in a slice.
      wptr = d_weights[ilayer] + idepth ; // First filter weight for this slice is here; subsequent weights spaced by wt_cols
      }

   sum = 0.0 ;

   // Center of first filter is at HalfWidth-Pad; filter begins at -Pad.
   rbase = rstart = d_strideV[ilayer] * iheight - d_padV[ilayer] ;
   rstop = rstart + 2 * d_HalfWidV[ilayer] ;
   cbase = cstart = d_strideH[ilayer] * iwidth - d_padH[ilayer] ;
   cstop = cstart + 2 * d_HalfWidH[ilayer] ;

   if (rstart < 0)
      rstart = 0 ;
   if (cstart < 0)
      cstart = 0 ;

   // It's annoying and messy, but we must duplicate the same code for the case of this being the
   // first hidden layer (fed by the input) versus a subsequent hidden layer (fed by prior activations).
   // This is because the input uses a float pointer, and activations a double pointer.
   // Deciding in the inner loop would be too slow!

   if (ilayer == 0) {
      f_inptr = d_predictors + (blockIdx.z + case_offset + case_start) * d_n_pred ;
      if (rstop >= d_img_rows)
         rstop = d_img_rows - 1 ;
      if (cstop >= d_img_cols)
         cstop = d_img_cols - 1 ;

      for (in_row=rstart ; in_row<=rstop ; in_row++) {
         kwt = (in_row - rbase) * nH ;
         kin = in_row*d_img_cols ;
         for (in_col=cstart ; in_col<=cstop ; in_col++) {
            wtsub = (kwt + in_col - cbase) * d_img_bands ;
            insub = (kin+in_col) * d_img_bands ;
            for (in_slice=0 ; in_slice<d_img_bands ; in_slice++) {
               // wtsub = ((in_row - rbase) * nH + in_col - cbase) * d_img_bands + in_slice ;
               // insub = (in_row*d_img_cols+in_col)*d_img_bands+in_slice ;
               sum += f_inptr[insub] * wptr[wtsub*wt_cols] ;
               ++wtsub ;
               ++insub ;
               } // For in_slice
            } // For in_col
         } // For in_row

      sum += wptr[(d_n_prior_weights[ilayer]-1) * wt_cols] ;      // Bias
      }

   else {
      actptr = d_act[ilayer-1] + (blockIdx.z + case_offset) * d_nhid[ilayer-1] ;
      n_height = d_height[ilayer-1] ;
      n_width = d_width[ilayer-1] ;
      n_depth = d_depth[ilayer-1] ;
      if (rstop >= n_height)
         rstop = n_height - 1 ;
      if (cstop >= n_width)
         cstop = n_width - 1 ;

      for (in_row=rstart ; in_row<=rstop ; in_row++) {
         kwt = (in_row - rbase) * nH ;
         kin = in_row*n_width ;
         for (in_col=cstart ; in_col<=cstop ; in_col++) {
            wtsub = (kwt + in_col - cbase) * n_depth ;
            insub = (kin+in_col) * n_depth ;
            for (in_slice=0 ; in_slice<d_depth[ilayer-1] ; in_slice++) {
               // wtsub = ((in_row - rbase) * nH + in_col - cbase) * n_depth + in_slice ;
               // insub = (in_row*n_width+in_col)*n_depth+in_slice ;
               sum += actptr[insub] * wptr[wtsub*wt_cols] ;
               ++wtsub ;
               ++insub ;
               } // For in_slice
            } // For in_col
         } // For in_row

      sum += wptr[(d_n_prior_weights[ilayer]-1) * wt_cols] ;      // Bias
      }

   if (sum > MAX_EXP)
      sum = 1.0 ;
   else {
      sum = exp ( 2.0 * sum ) ;
      sum = (sum - 1.0) / (sum + 1.0) ;
      }

   n_height = d_height[ilayer] ;
   n_width = d_width[ilayer] ;
   n_depth = d_depth[ilayer] ;
   actptr = d_act[ilayer] ;
   ihid = (iheight * n_width + iwidth) * n_depth + idepth ;   // Activity for any layer type is (height, width, depth)
   actptr[(blockIdx.z+case_offset)*d_nhid[ilayer]+ihid] = sum ;
}


int cuda_hidden_activation_LOCAL_CONV (
   int local_vs_conv , // Is this a LOCAL (vs CONV) layer?
   int istart ,        // First case in this batch
   int istop ,         // One past last case
   int nhid ,          // Number of hidden neurons in this layer
   int n_slices ,      // Depth of this layer
   int ilayer          // Layer to process
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   // nhid = height * width * depth
   assert ( nhid % n_slices == 0 ) ;
   assert ( nhid / n_slices <= 65535 ) ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_slices + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (n_slices + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nhid / n_slices ;   // Height times width; visual field size
   block_launch.z = istop - istart ;

   device_hidden_activation_LOCAL_CONV <<< block_launch , threads_per_block >>>
                                         ( local_vs_conv , istart , 0 , 0 , n_slices , ilayer ) ;   

   // This does not trigger an escape, but it keeps the message queue running
   user_pressed_escape () ;

   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_hidden_activation_LOCAL_CONV launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


__global__ void device_hidden_activation_LOCAL_CONV_shared (
   int local_vs_conv , // Is this a LOCAL (vs CONV) layer?
   int istart ,        // First case in this batch
   int ilayer          // Layer to process
   )
{
   int k, iheight, iwidth, idepth, icase, n_height, n_width, n_depth, wt_cols ;
   int ihid, inner, n_inner, inner_blocks, prod ;
   int rstart, rstop, cstart, cstop, rbase, cbase, in_slice, in_row, in_col, isub, nH ;
   float *f_inptr, *wptr ;
   double value, sum, *actptr ;

   // In a block, threadIdx.x and threadIdx.y are the location within the BLOCK_SIZE square block.
   // The entire matrix of cases (rows) by slices (column) is divided into these blocks,
   // each of which is a launched block whose location in the entire matrix is given by blockIdx.x and blockIdx.y.
   // The sharing logic ignores blockIdx.z, which is just the location in the visual field.
   // The next four quantities identify the location within the entire matrix.

   idepth = blockIdx.x * BLOCK_SIZE + threadIdx.x ;
   icase = blockIdx.y * BLOCK_SIZE + threadIdx.y ;

   iheight = blockIdx.z / d_width[ilayer] ;
   iwidth = blockIdx.z % d_width[ilayer] ;

   nH = 2 * d_HalfWidH[ilayer] + 1 ;  // Horizontal width of the filter

   // This thread will compute the activation of neuron (iheight, iwidth, idepth) for case icase.
   // However, the first step is for the threads in this block to cooperatively do the global
   // loads into shared memory of the weights and inputs relevant to this block.
   // We do this in a loop which covers the 'inner' (n_inner) dimension of the matrix multiplication.

   // Note that it is critical that idepth be associated with the thread.
   // This ensures that adjacent threads reference the same input, which allows efficient memory use.
   // Also, the weights are ordered so that depth-fastest changes produce perfect or very good coalescing.
   // Thus, neuron layout in current layer is (row, column, slice).
   // This gives strong motivation for LOCAL layers to have depth a multiple of 32.
   // To see why, note the ihid= below.  That multiplication ensures perfect coalescing of the weight fetches.

   if (local_vs_conv) {
      wt_cols = d_nhid_cols[ilayer] ; // Padded size of weight matrix rows; each has nhid data values, then zero padding
                                      // There are n_prior_weights rows (prior depth * (2*HalfWidH+1) * (2*HalfWidV+1) + 1)
      ihid = (iheight * d_width[ilayer] + iwidth) * d_depth[ilayer] + idepth ;
      wptr = d_weights[ilayer] + ihid ;  // Device weights are transpose of host weights, with this neuron changing fastest
      }                                  // Order is (height, width, depth)

   else {
      wt_cols = d_depth_cols[ilayer] ; // Padded size of weight matrix rows; each has depth[ilayer] data values, then zero padding
                                       // There are n_prior_weights rows (prior depth * (2*HalfWidH+1) * (2*HalfWidV+1) + 1)
                                       // A convolutional layer has a different weight set for each slice,
                                       // but the same weight set for all neurons (visual field placement) in a slice.
      wptr = d_weights[ilayer] + idepth ; // First filter weight for this slice is here; subsequent weights spaced by wt_cols
      }


   // Get a pointer to and the size of the prior-layer feeding this layer

   if (ilayer == 0) {
      f_inptr = d_predictors + (icase + istart) * d_n_pred ;
      n_height = d_img_rows ;
      n_width = d_img_cols ;
      n_depth = d_img_bands ;
      }

   else {
      actptr = d_act[ilayer-1] + icase * d_nhid[ilayer-1] ;
      n_height = d_height[ilayer-1] ;
      n_width = d_width[ilayer-1] ;
      n_depth = d_depth[ilayer-1] ;
      }

   // Center of first filter is at HalfWidth-Pad; filter begins at -Pad.
   // These quantities are independent of the depth (column here) and case (row here).

   rbase = rstart = d_strideV[ilayer] * iheight - d_padV[ilayer] ;
   rstop = rstart + 2 * d_HalfWidV[ilayer] ;
   cbase = cstart = d_strideH[ilayer] * iwidth - d_padH[ilayer] ;
   cstop = cstart + 2 * d_HalfWidH[ilayer] ;

   if (rstart < 0)
      rstart = 0 ;
   if (cstart < 0)
      cstart = 0 ;

   if (rstop >= n_height)
      rstop = n_height - 1 ;
   if (cstop >= n_width)
      cstop = n_width - 1 ;

   // The prep work is done.  We now cooperatively do the global load.
   // This thread will handle Row threadIdx.y, Column threadIdx.x of the BLOCK_SIZE square block
   // in a loop over all inner blocks.
   // In each pass, we start by computing the ordinal position in the filter dot product loop.

   prod = (cstop-cstart+1) * n_depth ;      // Each prior-layer row has this many elements
   n_inner = (rstop-rstart+1) * prod + 1 ;  // This many terms in inner sum (+1 is for bias)
   inner_blocks = (n_inner + BLOCK_SIZE - 1) / BLOCK_SIZE ;  // We will process this many 'inner' blocks

   sum = 0.0 ;

   for (inner=0 ; inner<inner_blocks ; inner++) {
      __shared__ double s_cases[BLOCK_SIZE][BLOCK_SIZE] ;
      __shared__ float s_slices[BLOCK_SIZE][BLOCK_SIZE] ;

      // Slice; We will sum over FIRST index (y) of s_slices
      isub = inner * BLOCK_SIZE + threadIdx.y ;   // Ordinal position in filter dot product loop
      if (isub >= n_inner)          // Outside inner block
         value = 0.0 ;
      else if (isub == n_inner-1)   // Bias
         value = wptr[(d_n_prior_weights[ilayer]-1) * wt_cols] ;
      else {
         in_row = isub / prod ;
         k = isub - in_row * prod ;
         in_col = k / n_depth ;
         in_slice = k % n_depth ;
         in_row += rstart ;
         in_col += cstart ;
         isub = ((in_row - rbase) * nH + in_col - cbase) * n_depth + in_slice ;
         value = wptr[isub*wt_cols] ;
         }
      s_slices[threadIdx.y][threadIdx.x] = value ;

      // Case; We will sum over SECOND index (x) of s_cases
      isub = inner * BLOCK_SIZE + threadIdx.x ;   // Ordinal position in filter dot product loop
      if (isub >= n_inner)          // Outside inner block
         value = 0.0 ;
      else if (isub == n_inner-1)   // Bias
         value = 1.0 ;
      else {
         in_row = isub / prod ;
         k = isub - in_row * prod ;
         in_col = k / n_depth ;
         in_slice = k % n_depth ;
         in_row += rstart ;
         in_col += cstart ;
         isub = (in_row*n_width+in_col)*n_depth+in_slice ;
         if (ilayer == 0)
            value = f_inptr[isub] ;
         else
            value = actptr[isub] ;
         }
      s_cases[threadIdx.y][threadIdx.x] = value ;

      __syncthreads () ;

      for (k=0 ; k<BLOCK_SIZE ; k++)
         sum += s_cases[threadIdx.y][k] * s_slices[k][threadIdx.x] ;

      __syncthreads () ;

      } // For inner


   if (sum > MAX_EXP)
      sum = 1.0 ;
   else {
      sum = exp ( 2.0 * sum ) ;
      sum = (sum - 1.0) / (sum + 1.0) ;
      }

   n_width = d_width[ilayer] ;
   n_depth = d_depth[ilayer] ;
   actptr = d_act[ilayer] ;
   ihid = (iheight * n_width + iwidth) * n_depth + idepth ;   // Activity for any layer type is (height, width, depth)
   actptr[icase*d_nhid[ilayer]+ihid] = sum ; // Perfectly coalesced if depth and nhid multiples of 32
}


int cuda_hidden_activation_LOCAL_CONV_shared (
   int local_vs_conv , // Is this a LOCAL (vs CONV) layer?
   int istart ,        // First case in this batch
   int istop ,         // One past last case
   int nhid ,          // Number of hidden neurons in this layer
   int n_slices ,      // Depth of this layer
   int ilayer          // Layer to process
   )
{
   int nc, warpsize, threads_per_block ;
   char msg[256] ;
   dim3 thread_launch, block_launch ;
   cudaError_t error_id ;

   // nhid = height * width * depth
   assert ( nhid % n_slices == 0 ) ;
   assert ( nhid / n_slices <= 65535 ) ;

/*
   If possible (it normally would be), handle as much as possible with the more efficient shared-memory method.
   But if not, just use the non-shared method.
*/

   nc = istop - istart ;

   if (n_slices < BLOCK_SIZE  ||  nc < BLOCK_SIZE)
      return cuda_hidden_activation_LOCAL_CONV ( local_vs_conv , istart , istop , nhid , n_slices , ilayer ) ;

   thread_launch.x = BLOCK_SIZE ;
   thread_launch.y = BLOCK_SIZE ;
   thread_launch.z = 1 ;

   block_launch.x = n_slices / BLOCK_SIZE ;
   block_launch.y = nc / BLOCK_SIZE ;
   block_launch.z = nhid / n_slices ;   // Height times width; visual field size

   device_hidden_activation_LOCAL_CONV_shared <<< block_launch , thread_launch >>> ( local_vs_conv , istart , ilayer ) ;   

   // This does not trigger an escape, but it keeps the message queue running
   user_pressed_escape () ;

   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_hidden_activation_LOCAL_CONV_shared launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }


/*
   Clean up any extra slices
*/

   if (n_slices % BLOCK_SIZE) {
      threads_per_block = n_slices % BLOCK_SIZE ;
      block_launch.x = 1 ;
      block_launch.y = nhid / n_slices ;   // Height times width; visual field size
      block_launch.z = nc ;

      device_hidden_activation_LOCAL_CONV <<< block_launch , threads_per_block >>>
                           ( local_vs_conv , istart , 0 , n_slices / BLOCK_SIZE * BLOCK_SIZE , n_slices % BLOCK_SIZE , ilayer ) ;   

      // This does not trigger an escape, but it keeps the message queue running
      user_pressed_escape () ;

      cudaDeviceSynchronize() ;
      error_id = cudaGetLastError () ;
      if (error_id != cudaSuccess) {
         sprintf_s ( msg , 255 , "cuda_hidden_activation_LOCAL_CONV launch (shared 1) error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         audit ( msg ) ;
         MEMTEXT ( msg ) ;
         return 1 ;
         }
      }


/*
   Clean up any extra cases
*/

   if (nc % BLOCK_SIZE) {
      warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future
      threads_per_block = (n_slices / BLOCK_SIZE * BLOCK_SIZE + warpsize - 1) / warpsize * warpsize ;
      if (threads_per_block > 4 * warpsize)
         threads_per_block = 4 * warpsize ;

      block_launch.x = (n_slices / BLOCK_SIZE * BLOCK_SIZE + threads_per_block - 1) / threads_per_block ;
      block_launch.y = nhid / n_slices ;   // Height times width; visual field size
      block_launch.z = nc % BLOCK_SIZE ;

      device_hidden_activation_LOCAL_CONV <<< block_launch , threads_per_block >>>
                  ( local_vs_conv , istart, nc / BLOCK_SIZE * BLOCK_SIZE , 0 , n_slices / BLOCK_SIZE * BLOCK_SIZE , ilayer ) ;   

      // This does not trigger an escape, but it keeps the message queue running
      user_pressed_escape () ;

      cudaDeviceSynchronize() ;
      error_id = cudaGetLastError () ;
      if (error_id != cudaSuccess) {
         sprintf_s ( msg , 255 , "cuda_hidden_activation_LOCAL_CONV launch (shared 2) error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
         audit ( msg ) ;
         MEMTEXT ( msg ) ;
         return 1 ;
         }
      }
      

   return 0 ;
}


/*
---------------------------------------------------------------------------------

   hidden_activation_POOLED - Activations for a POOLAVG or POOLMAX hidden layer

---------------------------------------------------------------------------------
*/

__global__ void device_hidden_activation_POOLED (
   int avg_vs_max ,    // Is this a POOLAVG (vs POOLMAX) layer?
   int istart ,        // First case in this batch
   int ilayer          // Layer to process
   )
{
   int icase, iheight, iwidth, idepth, n_width, n_depth, ihid ;
   int rstart, rstop, cstart, cstop, in_row, in_col, *poolmax_id_ptr ;
   float *f_inptr ;
   double x, *actptr, value ;

   idepth = blockIdx.x * blockDim.x + threadIdx.x ;

   if (idepth >= d_depth[ilayer])
      return ;

   n_width = d_width[ilayer] ;
   n_depth = d_depth[ilayer] ;

   iheight = blockIdx.y / n_width ;
   iwidth = blockIdx.y % n_width ;
   ihid = (iheight * n_width + iwidth) * n_depth + idepth ;   // Activity for any layer type is (height, width, depth)

   // We are about to compute the activation of neuron (iheight, iwidth, idepth) in this layer.
   // Note that it is critical that idepth be associated with the thread.
   // This ensures that adjacent threads reference the same input, which allows efficient memory use.

   icase = blockIdx.z ;

   rstart = d_strideV[ilayer] * iheight ;
   rstop = rstart + d_PoolWidV[ilayer] - 1 ;
   cstart = d_strideH[ilayer] * iwidth ;
   cstop = cstart + d_PoolWidH[ilayer] - 1 ;

   // It's annoying and messy, but we must duplicate the same code for the case of this being the
   // first hidden layer (fed by the input) versus a subsequent hidden layer (fed by prior activations).
   // This is because the input uses a float pointer, and activations a double pointer.
   // Deciding in the inner loop would be too slow!

   if (ilayer == 0) {
      f_inptr = d_predictors + (icase + istart) * d_n_pred ;

      if (avg_vs_max) {
         value = 0.0 ;
         for (in_row=rstart ; in_row<=rstop ; in_row++) {
            for (in_col=cstart ; in_col<=cstop ; in_col++)
               value += f_inptr[(in_row*d_img_cols+in_col)*d_img_bands+idepth] ;
            } // For in_row
         value /= d_PoolWidV[ilayer] * d_PoolWidH[ilayer] ;
         }

      else {
         poolmax_id_ptr = &d_poolmax_id[ilayer][ihid] + icase * d_nhid[ilayer] ;
         value = -1.e60 ;
         for (in_row=rstart ; in_row<=rstop ; in_row++) {
            for (in_col=cstart ; in_col<=cstop ; in_col++) {
               x = f_inptr[(in_row*d_img_cols+in_col)*d_img_bands+idepth] ;
               if (x > value) {
                  value = x ;
                  *poolmax_id_ptr = in_row * d_img_cols + in_col ;  // Save id of max for backprop pass
                  }
               } // For in_col
            } // For in_row
         } // POOLMAX
      } // If first hidden layer

   else {
      actptr = d_act[ilayer-1] + icase * d_nhid[ilayer-1] ;
      n_width = d_width[ilayer-1] ;
      n_depth = d_depth[ilayer-1] ;

      if (avg_vs_max) {
         value = 0.0 ;
         for (in_row=rstart ; in_row<=rstop ; in_row++) {
            for (in_col=cstart ; in_col<=cstop ; in_col++)
               value += actptr[(in_row*n_width+in_col)*n_depth+idepth] ;
            } // For in_row
         value /= d_PoolWidV[ilayer] * d_PoolWidH[ilayer] ;
         }

      else {
         poolmax_id_ptr = &d_poolmax_id[ilayer][ihid] + icase * d_nhid[ilayer] ;
         value = -1.e60 ;
         for (in_row=rstart ; in_row<=rstop ; in_row++) {
            for (in_col=cstart ; in_col<=cstop ; in_col++) {
               x = actptr[(in_row*n_width+in_col)*n_depth+idepth] ;
               if (x > value) {
                  value = x ;
                  *poolmax_id_ptr = in_row * d_width[ilayer-1] + in_col ;  // Save id of max for backprop pass
                  }
               } // For in_col
            } // For in_row
         } // POOLMAX
      }

   actptr = d_act[ilayer] ;
   actptr[icase*d_nhid[ilayer]+ihid] = value ;
}


int cuda_hidden_activation_POOLED (
   int avg_vs_max ,    // Is this a POOLAVG (vs POOLMAX) layer?
   int istart ,        // First case in this batch
   int istop ,         // One past last case
   int nhid ,          // Number of hidden neurons in this layer
   int n_slices ,      // Depth of this layer
   int ilayer          // Layer to process
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   // nhid = height * width * depth
   assert ( nhid % n_slices == 0 ) ;
   assert ( nhid / n_slices <= 65535 ) ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_slices + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (n_slices + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nhid / n_slices ;   // Height times width; visual field size
   block_launch.z = istop - istart ;

   device_hidden_activation_POOLED <<< block_launch , threads_per_block >>> ( avg_vs_max , istart , ilayer ) ;   

   // This does not trigger an escape, but it keeps the message queue running
   user_pressed_escape () ;

   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_hidden_activation_POOLED launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
-----------------------------------------------------------------------------------

   output_activation_no_hidden - Compute activations for the output layer
                                 This version is for when there is no hidden layer.

-----------------------------------------------------------------------------------
*/

__global__ void device_output_activation_no_hidden (
   int istart        // First case in this batch
   )
{
   int icase, iout, i_input ;
   double sum ;
   float *wptr, *inptr ;

   iout = blockIdx.x * blockDim.x + threadIdx.x ;

   if (iout >= d_n_classes)
      return ;

   icase = blockIdx.y ;
   wptr = d_weights[0] + iout ; // Weights on device have current neuron changing fastest

   inptr = d_predictors + (icase + istart) * d_n_pred ;
   sum = 0.0 ;

   for (i_input=0 ; i_input<d_n_pred ; i_input++) {   // Weights are transpose of Host, with target changing fastest
      sum += *wptr * inptr[i_input] ;
      wptr += d_n_classes_cols ;
      }
   sum += *wptr ;  // Bias

   d_output[(icase+istart)*d_n_classes+iout] = sum ;
}

int cuda_output_activation_no_hidden (
   int istart ,    // First case in this batch
   int istop       // One past last case
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_classes + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (n_classes + threads_per_block - 1) / threads_per_block ;
   block_launch.y = istop - istart ;
   block_launch.z = 1 ;

   device_output_activation_no_hidden <<< block_launch , threads_per_block >>> ( istart ) ;   

   // This does not trigger an escape, but it keeps the message queue running
   user_pressed_escape () ;

   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_output_activation_no_hidden launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
-----------------------------------------------------------------------------------

   output_activation - Compute activations for the output layer
                       This version is for when there is at least one hidden layer.

-----------------------------------------------------------------------------------
*/

__global__ void device_output_activation (
   int istart        // First case in this batch; needed for output
   )
{
   int icase, iout, i_input, n_inputs ;
   double sum ;
   float *wptr ;
   double *inptr ;

   iout = blockIdx.x * blockDim.x + threadIdx.x ;

   if (iout >= d_n_classes)
      return ;

   icase = blockIdx.y ;                  // Activities are zero origin, not offset by istart
   wptr = d_weights[d_n_layers] + iout ; // Weights on device have current neuron changing fastest
   n_inputs = d_nhid[d_n_layers-1] ;
   inptr = d_act[d_n_layers-1] + icase * n_inputs ;

   sum = 0.0 ;

   for (i_input=0 ; i_input<n_inputs ; i_input++) {   // Weights are transpose of Host, with target changing fastest
      sum += *wptr * inptr[i_input] ;
      wptr += d_n_classes_cols ;
      }
   sum += *wptr ;  // Bias

   d_output[(icase+istart)*d_n_classes+iout] = sum ;
}

int cuda_output_activation (
   int istart ,    // First case in this batch
   int istop       // One past last case
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_classes + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (n_classes + threads_per_block - 1) / threads_per_block ;
   block_launch.y = istop - istart ;
   block_launch.z = 1 ;

   device_output_activation <<< block_launch , threads_per_block >>> ( istart ) ;   

   // This does not trigger an escape, but it keeps the message queue running
   user_pressed_escape () ;

   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_output_activation launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   softmax - Do SoftMax modification of outputs for a batch

--------------------------------------------------------------------------------
*/

__global__ void device_softmax (
   int istart ,       // First case in this batch
   int istop          // One past last case
   )
{
   int icase, iout ;
   double *outptr, sum ;

   icase = blockIdx.x * blockDim.x + threadIdx.x ;

   if (icase >= istop - istart)
      return ;

   outptr = d_output + (icase + istart) * d_n_classes ;  // Output vector for this case
   sum = 0.0 ;

   for (iout=0 ; iout<d_n_classes ; iout++) {
      if (outptr[iout] < MAX_EXP)
         outptr[iout] = exp ( outptr[iout] ) ;
      else
         outptr[iout] = exp ( MAX_EXP ) ;
      sum += outptr[iout] ;
      }

   for (iout=0 ; iout<d_n_classes ; iout++)
      outptr[iout] /= sum ;
}


int cuda_softmax (
   int istart ,       // First case in this batch
   int istop          // One past last case
   )
{
   int n, warpsize, blocks_per_grid, threads_per_block ;
   char msg[256] ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   n = istop - istart ;   // Number of elements

   threads_per_block = (n + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   blocks_per_grid = (n + threads_per_block - 1) / threads_per_block ;

   device_softmax <<< blocks_per_grid , threads_per_block >>> ( istart , istop ) ;   

   // This does not trigger an escape, but it keeps the message queue running
   user_pressed_escape () ;

   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_cpx_softmax launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
------------------------------------------------------------------------------------------------

   cuda_cpx_ll - Given output activations and true classes, compute log likelihood
             This would be called after the entire training set is processed,
             not in batches.
             
------------------------------------------------------------------------------------------------
*/

__global__ void device_ll ()
{
   __shared__ double partial_ll[REDUC_THREADS] ;
   int i, n, n_classes, index ;
   double sum_ll ;

   index = threadIdx.x ;
   n = d_ncases ;
   n_classes = d_n_classes ;

   sum_ll = 0.0 ;   
   for (i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x)
      sum_ll -= log ( d_output[i*n_classes+d_class[i]] + 1.e-30 ) ;

   partial_ll[index] = sum_ll ;
   __syncthreads() ;

   for (i=blockDim.x>>1 ; i ; i>>=1) {
      if (index < i)
         partial_ll[index] += partial_ll[index+i] ;
      __syncthreads() ;
      }

   if (index == 0)
      d_ll_out[blockIdx.x] = partial_ll[0] ;
}


int cuda_ll (
   int n ,          // Number of values; n_cases
   double *ll       // Computed log likelihood
   )
{
   int i, blocks_per_grid ;
   double sum ;
   char msg[256] ;
   cudaError_t error_id ;

   blocks_per_grid = (n + REDUC_THREADS - 1) / REDUC_THREADS ;
   if (blocks_per_grid > REDUC_BLOCKS)
      blocks_per_grid = REDUC_BLOCKS ;

   device_ll <<< blocks_per_grid , REDUC_THREADS >>> () ;   

   // This does not trigger an escape, but it keeps the message queue running
   user_pressed_escape () ;

   cudaDeviceSynchronize() ;

   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_cpx_ll launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   error_id = cudaMemcpy ( reduc_fdata , h_ll_out , blocks_per_grid * sizeof(float) , cudaMemcpyDeviceToHost ) ;

   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_cpx_ll Memcpy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   sum = 0.0 ;
   for (i=0 ; i<blocks_per_grid ; i++)
      sum += reduc_fdata[i] ;
   *ll = sum ;

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   output_delta - Put output delta into this_delta

--------------------------------------------------------------------------------
*/

__global__ void device_output_delta (
   int istart       // First case in this batch
   )
{
   int icase, iout ;
   double target ;

   iout = blockIdx.x * blockDim.x + threadIdx.x ;

   if (iout >= d_n_classes)
      return ;

   icase = blockIdx.y ;
   target = (iout == d_class[istart+icase]) ? 1.0 : 0.0 ;

   // The output matrix has all training cases, hence we add istart, but delta is relative to this batch.
   d_this_delta[icase*d_n_classes+iout] = target - d_output[(istart+icase)*d_n_classes+iout] ;
}

int cuda_output_delta (
   int istart ,      // First case in this batch
   int istop ,       // One past last case
   int ntarg         // Number of targets (outputs, classes)
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (ntarg + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (ntarg + threads_per_block - 1) / threads_per_block ;
   block_launch.y = istop - istart ;
   block_launch.z = 1 ;

   device_output_delta <<< block_launch , threads_per_block >>> ( istart ) ;   

   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_output_delta launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   output_gradient - Compute output layer gradient

--------------------------------------------------------------------------------
*/

__global__ void device_output_gradient_no_hidden (
   int istart ,   // Index of first case in this batch
   int nc         // Number of cases in batch
   )
{
   int icase, iin ;
   float *gptr ;
   double input ;

   iin = blockIdx.x * blockDim.x + threadIdx.x ;

   if (iin > d_n_pred)
      return ;

   icase = blockIdx.y ;

   if (iin < d_n_pred)
      input = d_predictors[(istart+icase)*d_n_pred+iin] ;
   else
      input = 1.0 ;              // Bias

   // iout = blockIdx.z ; We directly use this below

   gptr = d_grad[0] + icase * d_n_weights ; // Gradient of output layer
   gptr[blockIdx.z*(d_n_pred+1)+iin] = d_this_delta[icase*d_n_classes+blockIdx.z] * input ;
}


__global__ void device_output_gradient (
   int nc ,        // Number of cases in batch
   int ilayer      // Hidden layer which feeds the output layer
   )
{
   int icase, ihid, nhid ;
   float *gptr ;
   double input ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;
   nhid = d_nhid[ilayer] ;    // Neurons in last hidden layer
   if (ihid > nhid)
      return ;

   icase = blockIdx.y ;

   if (ihid < nhid)
      input = d_act[ilayer][icase*nhid+ihid] ;
   else
      input = 1.0 ;              // Bias

   // iout = blockIdx.z ; We directly use this below

   gptr = d_grad[ilayer+1] + icase * d_n_weights ; // Gradient of output layer
   gptr[blockIdx.z*(nhid+1)+ihid] = d_this_delta[icase*d_n_classes+blockIdx.z] * input ;
}


int cuda_output_gradient (
   int istart ,    // Index of first case in this batch
   int nc ,        // Number of cases in batch
   int nin ,       // Number of inputs to last layer
   int ilayer ,    // Hidden layer which feeds the output layer
   int ntarg       // Number of targets (outputs, classes)
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nin + 1 + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nin + 1 + threads_per_block - 1) / threads_per_block ; // Include bias
   block_launch.y = nc ;
   block_launch.z = ntarg ;

   if (ilayer < 0)
      device_output_gradient_no_hidden <<< block_launch , threads_per_block >>> ( istart , nc ) ;   
   else
      device_output_gradient <<< block_launch , threads_per_block >>> ( nc , ilayer ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_output_gradient launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
-----------------------------------------------------------------------------------

   backprop_delta_FC - Backpropagate delta from a fully connected hidden layer

-----------------------------------------------------------------------------------
*/

__global__ void device_backprop_delta_FC (
   int ilayer     // Feed is from ilayer to ilayer+1, so ilayer+1 is FC
   )
{
   int j, icase, ihid, nhid, n_next ;
   float *next_weights ;
   double *delta_ptr, *prior_delta_ptr, this_act, delta ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;
   nhid = d_nhid[ilayer] ;       // Neurons in this hidden layer

   if (ihid >= nhid)
      return ;

   icase = blockIdx.y ;

   if (ilayer == d_n_layers-1) {
      n_next = d_n_classes ;
      next_weights = d_weights[ilayer+1] + ihid * d_n_classes_cols ;
      }
   else {
      n_next = d_nhid[ilayer+1] ;
      next_weights = d_weights[ilayer+1] + ihid * d_nhid_cols[ilayer+1] ;
      }

   delta_ptr = d_this_delta + icase * n_next ;      // Coming from the next layer, which was just done
   prior_delta_ptr = d_prior_delta + icase * nhid ; // Save for the next layer to do, one layer back

   delta = 0.0 ;
   for (j=0 ; j<n_next ; j++)
      delta += delta_ptr[j] * next_weights[j] ;     // Weights are transpose of host; later layer changes fastest

   if (d_layer_type[ilayer] == TYPE_FC  || d_layer_type[ilayer] == TYPE_LOCAL || d_layer_type[ilayer] == TYPE_CONV) {
      this_act = d_act[ilayer][icase*nhid+ihid] ;
      delta *= 1.0 - this_act * this_act ;          // Derivative
      }

   prior_delta_ptr[ihid] = delta ;                  // Save it for doing the next layer back
}

int cuda_backprop_delta_FC (
   int nc ,           // Number of cases in batch
   int ilayer ,       // Hidden layer being processed
   int nhid_this      // Number of hidden neurons in this layer
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid_this + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nhid_this + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_backprop_delta_FC <<< block_launch , threads_per_block >>> ( ilayer ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_backprop_delta_FC launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
-----------------------------------------------------------------------------------

   backprop_delta_nonpooled - Backpropagate delta from a locally connected
                              or convolutional hidden layer

-----------------------------------------------------------------------------------
*/

__global__ void device_backprop_delta_nonpooled (
   int ilayer     // Feed is from ilayer to ilayer+1, so ilayer+1 is LOCAL or CONV
   )
{
   int k, icase, ihid, next_row, next_col, next_slice, this_row, this_col, this_slice ;
   int nH, k_next, wt_cols, rstart, cstart, prod, ltype ;
   int strideH, strideV, padH, padV, height, width, depth ;
   int next_rstart, next_rstop, next_cstart, next_cstop ;
   float *weights, *wtptr ;
   double *this_delta_ptr, *prior_delta_ptr ;
   double this_act, sum ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;

   if (ihid >= d_nhid[ilayer])
      return ;

   prod = d_width[ilayer] * d_depth[ilayer] ; // Get the 3D coordinates of neuron 'ihid'
   this_row = ihid / prod ;
   k = ihid - this_row * prod ;
   this_col = k / d_depth[ilayer] ;
   this_slice = k % d_depth[ilayer] ;

   icase = blockIdx.y ;

   nH = 2 * d_HalfWidH[ilayer+1] + 1 ;  // Horizontal filter size

   this_delta_ptr = d_this_delta + icase * d_nhid[ilayer+1] ; // Coming from the next layer, which was just done
   prior_delta_ptr = d_prior_delta + icase * d_nhid[ilayer] ; // Save for the next layer to do, one layer back

   ltype = d_layer_type[ilayer+1] ;
   strideV = d_strideV[ilayer+1] ;
   strideH = d_strideH[ilayer+1] ;
   padV = d_padV[ilayer+1] ;
   padH = d_padH[ilayer+1] ;
   height = d_height[ilayer+1] ;
   width = d_width[ilayer+1] ;
   depth = d_depth[ilayer+1] ;

   // this >= next * stride - pad  IMPLIES  next <= (this + pad) / stride
   // this <= next * stride - pad + 2 * hw  IMPLIES  next >= (this + pad - 2 * hw) / stride
   // We can safely do this in integer arithmetic

   next_rstop = this_row + padV ;
   k = next_rstart = next_rstop - 2 * d_HalfWidV[ilayer+1] ;
   next_rstop /= strideV ;
   next_rstart /= strideV ;
   if (k >= 0  &&  k % strideV)  // If the division above was inexact,
      ++next_rstart ;            // we must by pass the fractional part
   if (next_rstop >= height)
      next_rstop = height - 1 ;
   if (next_rstart < 0)
      next_rstart = 0 ;

   next_cstop = this_col + padH ;
   k = next_cstart = next_cstop - 2 * d_HalfWidH[ilayer+1] ;
   next_cstop /= strideH ;
   next_cstart /= strideH ;
   if (k >= 0  &&  k % strideH)
      ++next_cstart ;
   if (next_cstop >= width)
      next_cstop = width - 1 ;
   if (next_cstart < 0)
      next_cstart = 0 ;

   weights = d_weights[ilayer+1] ;
   if (ltype == TYPE_CONV)  // A CONV layer has the same weight set for all neurons in visible field
      wt_cols = d_depth_cols[ilayer+1] ;
   else                     // A LOCAL layer has different weights for each neuron
      wt_cols = d_nhid_cols[ilayer+1] ; // For LOCAL layers, neuron layout in current layer is (height, width, depth).

   sum = 0.0 ;

   for (next_row=next_rstart ; next_row<=next_rstop ; next_row++) {
      for (next_col=next_cstart ; next_col<=next_cstop ; next_col++) {

         // Center of first filter is at HalfWidth-Pad; filter begins at -Pad.
         rstart = strideV * next_row - padV ;
         cstart = strideH * next_col - padH ;


         // This is what we would be testing if we didn't compute the exact limits above
         // rstop = rstart + 2 * d_HalfWidV[ilayer+1] ;
         // cstop = cstart + 2 * d_HalfWidH[ilayer+1] ;
         // if (this_row >= rstart  &&  this_row <= rstop  &&  this_col >= cstart  &&  this_col <= cstop) {

         for (next_slice=0 ; next_slice<depth ; next_slice++) {
            k_next = (next_row * width + next_col) * depth + next_slice ;
            if (ltype == TYPE_CONV)             // A CONV layer has the same weight set for all neurons in visible field
               wtptr = weights + next_slice ;
            else                                // A LOCAL layer has different weights for each neuron (height, width, depth)
               wtptr = weights + k_next ;
            k = ((this_row - rstart) * nH + this_col - cstart) * d_depth[ilayer] + this_slice ; // Location in filter
            sum += this_delta_ptr[k_next] * wtptr[k*wt_cols] ;
            }  // For next_col
         }  // For next_row
      }  // For next_slice

//   ihid = (this_row * d_width[ilayer] + this_col) * d_depth[ilayer] + this_slice ;
   if (d_layer_type[ilayer] == TYPE_FC  || d_layer_type[ilayer] == TYPE_LOCAL || d_layer_type[ilayer] == TYPE_CONV) {
      this_act = d_act[ilayer][icase*d_nhid[ilayer]+ihid] ;
      sum *= 1.0 - this_act * this_act ;             // Derivative
      }
   prior_delta_ptr[ihid] = sum ;                  // Save it for doing the next layer back
}

int cuda_backprop_delta_nonpooled (
   int nc ,           // Number of cases in batch
   int ilayer ,       // Hidden layer being processed
   int nhid_this      // Number of hidden neurons in this layer
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid_this + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nhid_this + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_backprop_delta_nonpooled <<< block_launch , threads_per_block >>> ( ilayer ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_backprop_delta_nonpooled launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
-----------------------------------------------------------------------------------

   backprop_delta_pooled - Backpropagate delta from a POOLAVG or POOLMAX layer

-----------------------------------------------------------------------------------
*/

__global__ void device_backprop_delta_pooled (
   int ilayer     // Feed is from ilayer to ilayer+1, so ilayer+1 is POOLAVG or POOLMAX
   )
{
   int k, icase, ihid, next_row, next_col, this_row, this_col, this_slice ;
   int k_next, prod, this_cols, *poolmax_id_ptr ;
   int next_rstart, next_rstop, next_cstart, next_cstop ;
   double *this_delta_ptr, *prior_delta_ptr, sum, this_act ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;

   if (ihid >= d_nhid[ilayer])
      return ;

   prod = d_width[ilayer] * d_depth[ilayer] ;
   this_row = ihid / prod ;
   k = ihid - this_row * prod ;
   this_col = k / d_depth[ilayer] ;
   this_slice = k % d_depth[ilayer] ;

   icase = blockIdx.y ;

   this_delta_ptr = d_this_delta + icase * d_nhid[ilayer+1] ; // Coming from the next layer, which was just done
   prior_delta_ptr = d_prior_delta + icase * d_nhid[ilayer] ; // Save for the next layer to do, one layer back

   // this >= next * stride  IMPLIES  next <= this / stride
   // this <= next * stride + pw - 1  IMPLIES  next >= (this - pw + 1) / stride
   // We can safely do this in integer arithmetic

   next_rstop = this_row ;
   k = next_rstart = next_rstop - d_PoolWidV[ilayer+1] + 1 ;
   next_rstop /= d_strideV[ilayer+1] ;
   next_rstart /= d_strideV[ilayer+1] ;
   if (k >= 0  &&  k % d_strideV[ilayer+1])
      ++next_rstart ;
   if (next_rstop >= d_height[ilayer+1])
      next_rstop = d_height[ilayer+1] - 1 ;
   if (next_rstart < 0)
      next_rstart = 0 ;

   next_cstop = this_col ;
   k = next_cstart = next_cstop - d_PoolWidH[ilayer+1] + 1 ;
   next_cstop /= d_strideH[ilayer+1] ;
   next_cstart /= d_strideH[ilayer+1] ;
   if (k >= 0  &&  k % d_strideH[ilayer+1])
      ++next_cstart ;
   if (next_cstop >= d_width[ilayer+1])
      next_cstop = d_width[ilayer+1] - 1 ;
   if (next_cstart < 0)
      next_cstart = 0 ;

   sum = 0.0 ;

   if (d_layer_type[ilayer+1] == TYPE_POOLAVG) {
      for (next_row=next_rstart ; next_row<=next_rstop ; next_row++) {
         for (next_col=next_cstart ; next_col<=next_cstop ; next_col++) {
            k_next = (next_row * d_width[ilayer+1] + next_col) * d_depth[ilayer+1] + this_slice ;
            sum += this_delta_ptr[k_next] ;
            }  // For next_col
         }  // For next_row
      sum /= d_PoolWidH[ilayer+1] * d_PoolWidV[ilayer+1] ;
      } // POOLAVG

   else if (d_layer_type[ilayer+1] == TYPE_POOLMAX) {
      poolmax_id_ptr = d_poolmax_id[ilayer+1] + icase * d_nhid[ilayer+1] ;
      this_cols = d_width[ilayer] ;
      for (next_row=next_rstart ; next_row<=next_rstop ; next_row++) {
         for (next_col=next_cstart ; next_col<=next_cstop ; next_col++) {
            k_next = (next_row * d_width[ilayer+1] + next_col) * d_depth[ilayer+1] + this_slice ;
            // Was the current-layer neuron the winner in the MAX competition for the next-layer competition?
            if (this_row == poolmax_id_ptr[k_next] / this_cols &&
                this_col == poolmax_id_ptr[k_next] % this_cols)
               sum += this_delta_ptr[k_next] ;  // Weight is 1
            }  // For next_col
         }  // For next_row
      } // POOLMAX

//   ihid = (this_row * d_width[ilayer] + this_col) * d_depth[ilayer] + this_slice ;
   if (d_layer_type[ilayer] == TYPE_FC  || d_layer_type[ilayer] == TYPE_LOCAL || d_layer_type[ilayer] == TYPE_CONV) {
      this_act = d_act[ilayer][icase*d_nhid[ilayer]+ihid] ;
      sum *= 1.0 - this_act * this_act ;             // Derivative
      }
   prior_delta_ptr[ihid] = sum ;                  // Save it for doing the next layer back
}

int cuda_backprop_delta_pooled (
   int nc ,           // Number of cases in batch
   int ilayer ,       // Hidden layer being processed
   int nhid_this      // Number of hidden neurons in this layer
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid_this + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nhid_this + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_backprop_delta_pooled <<< block_launch , threads_per_block >>> ( ilayer ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_backprop_delta_pooled launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
-----------------------------------------------------------------------------------

   move_delta - Move delta from prior_delta to this_delta

-----------------------------------------------------------------------------------
*/

__global__ void device_move_delta (
   int nhid      // Number of neurons in the layer just processed
   )
{
   int icase, ihid ;

   ihid = blockIdx.x * blockDim.x + threadIdx.x ;

   if (ihid >= nhid)
      return ;

   icase = blockIdx.y ;

   d_this_delta[icase*nhid+ihid] = d_prior_delta[icase*nhid+ihid] ;
}

int cuda_move_delta (
   int nc ,           // Number of cases in batch
   int nhid_this      // Number of hidden neurons in this layer
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (nhid_this + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (nhid_this + threads_per_block - 1) / threads_per_block ; // Include bias
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_move_delta <<< block_launch , threads_per_block >>> ( nhid_this ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_move_delta launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
-----------------------------------------------------------------------------------

   hidden_gradient_FC - Compute gradient for a fully connected hidden layer

-----------------------------------------------------------------------------------
*/

__global__ void device_hidden_gradient_FC (
   int istart ,    // Index of first case in this batch
   int nc ,        // Number of cases in batch
   int ilayer      // Hidden layer being processed
   )
{
   int iin, ihid, nin, ninp1 ;
   float *gptr ;
   double input ;

   iin = blockIdx.x * blockDim.x + threadIdx.x ;

   if (ilayer == 0)
      nin = d_n_pred ;          // Number of inputs to each neuron in this layer
   else
      nin = d_nhid[ilayer-1] ;

   // icase = blockIdx.z ;      // Used directly below

   if (iin > nin)
      return ;
   else if (iin == nin)
      input = 1.0 ;              // Bias
   else if (ilayer)
      input = d_act[ilayer-1][blockIdx.z*nin+iin] ;
   else
      input = d_predictors[(istart+blockIdx.z)*nin+iin] ;
   ihid = blockIdx.y ;
   ninp1 = nin + 1 ;             // We mustn't forget the bias, so nin+1

   gptr = d_grad[ilayer] + blockIdx.z * d_n_weights ;  // Gradient of this hidden layer for this case
   gptr[ihid*ninp1+iin] = d_this_delta[blockIdx.z*d_nhid[ilayer]+ihid] * input ;
}


/*
------------------------------------------------------------------------------------------------------

   hidden_gradient_LOCAL_CONV - Compute gradient for a locally connected or convolutional hidden layer

   For a LOCAL layer, we do all of the nhid * n_prior_weights * max_batch entries.
   But for a CONV layer, there are just depth * n_prior_weights * max_batch entries
   because the weight set for every (height,width) placement in the visual field is the same
   in any single slice.  So there are not enough entries in the gradient vector.
   Thus, we use the previously allocated convgrad_work vector, which has
   nhid * n_prior_weights * max_batch entries.
   Then we will launch another kernel which flattens out the height and width dimensions
   by summing them into the gradient vector.

   Note: ihidstart must be a multiple of height*width!

------------------------------------------------------------------------------------------------------
*/

__global__ void device_hidden_gradient_LOCAL_CONV (
   int local_vs_conv , // Is this a LOCAL (vs CONV) layer?
   int nfilt ,         // Filter size, (2*hwV+1) * (2*hwH+1) * depth of input (does not include +1 for bias)
   int istart ,        // Index of first case in this batch
   int depth_offset ,  // Start processing layers at this depth
   int n_depths ,      // Number of slices to be processed
   int ilayer          // Hidden layer being processed
   )
{
   int k, iin, ifilt, ihid_offset, ihid_actual, prod ;
   int in_row, in_col, in_slice, in_rows, in_cols, in_slices ;
   int this_row, this_col, ifiltV, ifiltH ;
   float *gptr ;
   double input, delta ;

   ifilt = blockIdx.x * blockDim.x + threadIdx.x ; // <= filter size
   if (ifilt > nfilt)
      return ;

   // Input is from either the input image or a prior layer's activations
   // Get the input dimensions (height, width, depth)

   if (ilayer == 0) {
      in_rows = d_img_rows ;
      in_cols = d_img_cols ;
      in_slices = d_img_bands ;
      }
   else {
      in_rows = d_height[ilayer-1] ;
      in_cols = d_width[ilayer-1] ;
      in_slices = d_depth[ilayer-1] ;
      }

   // We may be splitting the computation into multiple launches, doing one or more slices in each.
   // If so, we need to compute the actual slice/neuron being processed here.
   // If we are doing a CONV layer, the offset will be into convgrad_work.
   // Whenever we access data, we use ihid_actual, and we also use it to save a LOCAL gradient.
   // But when we save a CONV gradient, we use ihid_offset.
   // Recall that hidden neurons are stored with depth changing fastest.

   ihid_offset = blockIdx.y ;                       // Offset into this launch set
   prod = d_width[ilayer] * d_height[ilayer] ;      // Size of visual field, a slice
   k = ihid_offset % n_depths + depth_offset ;      // Actual starting slice
   ihid_actual = ihid_offset / n_depths * d_depth[ilayer] + k ; // Actual hidden neuron being done

   // If this is the bias term, it's simple.
   // Recall that blockIdx.z is the case in this batch

   if (ifilt == nfilt) {    // Bias term
      delta = d_this_delta[blockIdx.z*d_nhid[ilayer]+ihid_actual] ;
      if (local_vs_conv) {
         gptr = d_grad[ilayer] + blockIdx.z * d_n_weights ;  // Gradient of this hidden layer for this case
         gptr[ihid_actual*d_n_prior_weights[ilayer]+d_n_prior_weights[ilayer]-1] = delta ;
         }
      else {
         gptr = d_convgrad_work + blockIdx.z * d_max_convgrad_each ;
         gptr[ihid_offset*d_convgrad_cols[ilayer]+d_n_prior_weights[ilayer]-1] = delta ;
         }
      return ;
      }

   // Get the location of this kernel within the filter
   // Thread ifilt is the ordinal number of the filter element
   // The filter order is (height, width, slice)

   prod = (2 * d_HalfWidH[ilayer] + 1) * in_slices ;
   ifiltV = ifilt / prod ;
   k = ifilt - ifiltV * prod ;
   ifiltH = k / in_slices ;
   in_slice = k % in_slices ;

   // Get the location of this neuron within the volume of the current layer

   prod = d_width[ilayer] * d_depth[ilayer] ;
   this_row = ihid_actual / prod ;
   k = ihid_actual - this_row * prod ;
   this_col = k / d_depth[ilayer] ;
//   this_slice = k % d_depth[ilayer] ;  // Not needed; here for clarity only

   // Get the location of this filter element within the input volume.
   // It may be outside an edge, in which case there is nothing to do.
   // The filter center is at stride * CurrentPos + HalfWidth - Pad.
   // The upper-left corner is at stride * CurrentPos - Pad.
   // This can cause branch-induced stalling, but only at edges.

   in_row = d_strideV[ilayer] * this_row - d_padV[ilayer] + ifiltV ;
   if (in_row < 0  ||  in_row >= in_rows)
      return ;

   in_col = d_strideH[ilayer] * this_col - d_padH[ilayer] + ifiltH ;
   if (in_col < 0  ||  in_col >= in_cols)
      return ;

   // Here we go

   if (local_vs_conv)
      gptr = d_grad[ilayer] + blockIdx.z * d_n_weights ;  // Gradient of this hidden layer for this case
   else
      gptr = d_convgrad_work + blockIdx.z * d_max_convgrad_each ;
   delta = d_this_delta[blockIdx.z*d_nhid[ilayer]+ihid_actual] ;

   // Fetch the input.  Adjacent threads have adjacent memory accesses, though not zero padded for alignment.
   // But zero padding would do no good here because in general, warps will only by chance start with iin=0.
   // All is great if in_slices and prior-layer size are multiples of 16!

   iin = (in_row * in_cols + in_col) * in_slices + in_slice ;
   if (ilayer)
      input = d_act[ilayer-1][blockIdx.z*d_nhid[ilayer-1]+iin] ;
   else
      input = d_predictors[(istart+blockIdx.z)*d_n_pred+iin] ;

   // Adjacent threads access adjacent memory, though there is no zero padding for alignment.
   // Zero padding here would help, because ifilt starts at zero.
   // But that would complicate the code a lot, and this is a small fraction of instructions.
   // Also, the kernel is generally limited by the math pipeline.
   // And of course if n_prior_weights is a multiple of 32, all is good!

   if (local_vs_conv)
      gptr[ihid_actual*d_n_prior_weights[ilayer]+ifilt] = input * delta ;
   else
      gptr[ihid_offset*d_convgrad_cols[ilayer]+ifilt] = input * delta ;
}


__global__ void device_flatten_gradient (
   int islice_start ,  // Index of first slice in this batch
   int max_depth ,     // Max depth in launch, <= slices reserved in convgrad_work
   int ilayer          // Hidden layer being processed
   )
{
   int k, islice, icase, iprior, irow, icol ;
   double sum ;
   float *workptr, *gradptr ;

   iprior = blockIdx.x * blockDim.x + threadIdx.x ;
   if (iprior >= d_n_prior_weights[ilayer])
      return ;

   islice = blockIdx.y ;
   icase = blockIdx.z ;

   gradptr = d_grad[ilayer] + icase * d_n_weights ;  // Gradient of this hidden layer for this case
   workptr = d_convgrad_work + icase * d_max_convgrad_each ;

//   nvisual = d_height[ilayer] * d_width[ilayer] ;    // Also equals nhid / depth
   sum = 0.0 ;

   for (irow=0 ; irow<d_height[ilayer] ; irow++) {
      for (icol=0 ; icol<d_width[ilayer] ; icol++) {
         k = (irow * d_width[ilayer] + icol) * max_depth + islice ;  // The neuron at (irow, icol, islice)
//         assert ( k*d_convgrad_cols[ilayer]+iprior < d_max_convgrad_each ) ;
         sum += workptr[k*d_convgrad_cols[ilayer]+iprior] ;
         }
      }

   gradptr[(islice+islice_start)*d_n_prior_weights[ilayer]+iprior] = sum ;
}


int cuda_hidden_gradient (
   int max_hid_grad ,    // Max hid in a CONV hid grad launch; multiple of height*width; <= 65535
   int max_mem_grad ,    // Maximum CONV working memory (MB) per CUDA launch; prevents timeout error and lowers memory use
   int istart ,          // Index of first case in this batch
   int nc ,              // Number of cases in batch
   int ilayer ,          // Hidden layer being processed
   int type ,            // Type of this layer
   int nhid_this ,       // Number of hidden neurons in this layer
   int nhid_prior ,      // And in prior layer
   int depth ,           // Depth of this layer
   int n_prior_weights , // N of inputs per neuron (including bias) to prior layer = prior depth * (2*HalfWidH+1) * (2*HalfWidV+1) + 1
   int *n_launches       // Returned for user edification
   )
{
   int i, conv_cols, n_max, nhid_launch, ihid_start, warpsize, threads_per_block, field, divisor ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   field = nhid_this / depth ;           // Visual field size = height * width
   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   *n_launches = 1 ;                     // This is purely for reporting launch statistics

   if (type == TYPE_FC) {
      threads_per_block = (nhid_prior + 1 + warpsize - 1) / warpsize * warpsize ;  // +1 includes bias
      if (threads_per_block > 4 * warpsize)
         threads_per_block = 4 * warpsize ;
      block_launch.x = (nhid_prior + 1 + threads_per_block - 1) / threads_per_block ; // Include bias
      block_launch.y = nhid_this ;
      block_launch.z = nc ;
      device_hidden_gradient_FC <<< block_launch , threads_per_block >>> ( istart , nc , ilayer ) ;   
      cudaDeviceSynchronize() ;
      }


   else if (type == TYPE_LOCAL  ||  type == TYPE_CONV) {

      divisor = 1 ; // Figure out how much we have to divide slices to meet max_hid_grad and max_mem_grad limits
      if (type == TYPE_CONV) {
         conv_cols = (n_prior_weights + 31) / 32 * 32 ; // CONV scratch is zero padded for full coalescing
         n_max = 1024 * 1024 * max_mem_grad / (max_batch * conv_cols * sizeof(float)) ; // Launch limit satisfying memory
         }
      else
         n_max = MAXPOSNUM ;
      for ( ;; ) {
         nhid_launch = depth / divisor * field ; // We will launch this many hid at a time
         if (nhid_launch <= max_hid_grad  &&  nhid_launch <= n_max)
            break ;
         ++divisor ;
         }
      if (nhid_launch < field)   // Careless user may have set it too small
         nhid_launch = field ;   // So ignore it

/*
   Launch loop
*/

      *n_launches = 0 ;

      if (type == TYPE_CONV) {
         // We must zero the CONV work area because some entries may be undefined
         // This must also be done in the last pass, because a partial launch at the end
         // may have garbage from the prior launch in 'undefined' locations.
         for (i=0 ; i<max_convgrad_work ; i++)
            fdata[i] = 0.0 ;
         error_id = cudaMemcpy ( h_convgrad_work , fdata , max_convgrad_work * sizeof(float) , cudaMemcpyHostToDevice ) ;
         if (error_id != cudaSuccess) {
            sprintf_s ( msg , 255 , "cuda_hidden_gradient_LOCAL_CONV convgrad_work zero error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
            audit ( msg ) ;
            MEMTEXT ( msg ) ;
            return 1 ;
            }
         }

      for (ihid_start=0 ; ihid_start < depth*field ; ihid_start+=nhid_launch) {

         threads_per_block = (n_prior_weights + warpsize - 1) / warpsize * warpsize ;
         if (threads_per_block > 4 * warpsize)   // Increase?  May be reasonable
            threads_per_block = 4 * warpsize ;
         block_launch.x = (n_prior_weights + threads_per_block - 1) / threads_per_block ;

         block_launch.y = nhid_launch ;
         if (depth*field - ihid_start < nhid_launch) {  // Last launch may be partial
            block_launch.y = depth*field - ihid_start ;
            if (type == TYPE_CONV) {
               for (i=0 ; i<max_convgrad_work ; i++)
                  fdata[i] = 0.0 ;
               error_id = cudaMemcpy ( h_convgrad_work , fdata , max_convgrad_work * sizeof(float) , cudaMemcpyHostToDevice ) ;
               if (error_id != cudaSuccess) {
                  sprintf_s ( msg , 255 , "cuda_hidden_gradient_LOCAL_CONV convgrad_work zero error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
                  audit ( msg ) ;
                  MEMTEXT ( msg ) ;
                  return 1 ;
                  }
               }
            } // If last launch is partial

         block_launch.z = nc ;
         device_hidden_gradient_LOCAL_CONV <<< block_launch , threads_per_block >>>
                ( type==TYPE_LOCAL ? 1 : 0 , n_prior_weights-1 , istart , ihid_start/field , block_launch.y/field , ilayer ) ;   
         cudaDeviceSynchronize() ;
         error_id = cudaGetLastError () ;
         if (error_id != cudaSuccess) {
            sprintf_s ( msg , 255 , "cuda_hidden_gradient LOCAL_CONV launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
            audit ( msg ) ;
            MEMTEXT ( msg ) ;
            return 1 ;
            }

         if (type == TYPE_CONV) {  // Must also flatten gradient?
            assert ( nhid_launch * nc * n_prior_weights <= max_convgrad_work ) ;
            threads_per_block = (n_prior_weights + warpsize - 1) / warpsize * warpsize ;
            if (threads_per_block > 4 * warpsize)
               threads_per_block = 4 * warpsize ;

            block_launch.x = (n_prior_weights + threads_per_block - 1) / threads_per_block ;
            block_launch.y /= field ;  // Number of slices in launch
            block_launch.z = nc ;

            device_flatten_gradient <<< block_launch , threads_per_block >>> ( ihid_start/field , block_launch.y , ilayer ) ;   
            cudaDeviceSynchronize() ;
            error_id = cudaGetLastError () ;
            if (error_id != cudaSuccess) {
               sprintf_s ( msg , 255 , "cuda_hidden_gradient flatten_gradient launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
               audit ( msg ) ;
               MEMTEXT ( msg ) ;
               return 1 ;
               }
            } // CONV
         ++*n_launches ;
         } // Launch loop
      } // LOCAL or CONV

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   zero_gradient - Some gradient entires may be undefined (zero, actually)
                   due to lack of connections in a poorly designed model.
                   So before computing the gradient for a batch,
                   we must zero the entire gradient vector.

--------------------------------------------------------------------------------
*/

__global__ void device_zero_gradient (
   int nc          // Number of cases in batch
   )
{
   int index, icase ;
   float *gptr ;

   index = blockIdx.x * blockDim.x + threadIdx.x ;

   if (index >= d_n_weights)
      return ;

   icase = blockIdx.y ;

   gptr = d_grad[0] + index ;  // Complete gradient starts at [0]
   gptr[icase*d_n_weights] = 0.0f ;
}

int cuda_zero_gradient (
   int nc ,          // Number of cases in batch
   int n_weights     // Number of weights
   )
{
   int warpsize, threads_per_block ;
   char msg[256] ;
   dim3 block_launch ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_weights + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (n_weights + threads_per_block - 1) / threads_per_block ;
   block_launch.y = nc ;
   block_launch.z = 1 ;

   device_zero_gradient <<< block_launch , threads_per_block >>> ( nc ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_zero_gradient launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   fetch_gradient - Retrieve sum across batch of complete gradient

   The CUDA grad is neither the order of the CUDA weights, nor the HOST grad!
   Rather, they are grouped by current neuron, (row, col, slice), and with
   input order as the CUDA inputs (row, column, slice).

   A fully connected layer has height=width=1; all neurons are depth.

--------------------------------------------------------------------------------
*/

__global__ void device_fetch_gradient (
   int nc          // Number of cases in batch
   )
{
   int index, icase ;
   float *gptr ;
   double sum ;

   index = blockIdx.x * blockDim.x + threadIdx.x ;

   if (index >= d_n_weights)
      return ;

   sum = 0.0 ;
   gptr = d_grad[0] + index ;  // Complete gradient starts at [0]
   for (icase=0 ; icase<nc ; icase++)   // For all cases in this batch
      sum += gptr[icase*d_n_weights] ;
   *gptr = sum ;
}

int cuda_fetch_gradient (
   int nc ,            // Number of cases in batch
   int n_weights ,     // Number of weights
   double **hostgrad , // Gradient sum output here
   int n_classes ,     // Number of outputs
   int n_layers ,      // Hidden layers; does not include output
   int *layer_type ,   // Each entry (input to final) is TYPE_? in CONST.H
   int img_rows ,      // Size of input image
   int img_cols ,
   int img_bands ,
   int *height ,       // Height of visible field in each layer
   int *width ,        // Width of visible field
   int *depth ,        // Number of slices in each layer
   int *nhid ,         // Number of hidden neurons in each layer
   int *hwH ,          // Half-width of filters
   int *hwV
   )
{
   int warpsize, blocks_per_grid, threads_per_block ;
   int n, n_prior, ilayer, isub ;
   int idepth, iheight, iwidth, ndepth, nheight, nwidth ;
   int in_row, in_col, in_slice, in_n_height, in_n_width, in_n_depth ;
   double *gptr ;
   float *fptr ;
   char msg[256] ;
   cudaError_t error_id ;

   warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future

   threads_per_block = (n_weights + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   blocks_per_grid = (n_weights + threads_per_block - 1) / threads_per_block ;

   device_fetch_gradient <<< blocks_per_grid , threads_per_block >>> ( nc ) ;   
   cudaDeviceSynchronize() ;
   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_fetch_gradient launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

   error_id = cudaMemcpy ( fdata , grad , n_weights * sizeof(float) , cudaMemcpyDeviceToHost ) ;
   if (error_id != cudaSuccess) {
      sprintf_s ( msg , 255 , "cuda_fetch_gradient copy error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
      audit ( msg ) ;
      MEMTEXT ( msg ) ;
      return 1 ;
      }

/*
   Reorder
*/
   
   fptr = fdata ;

   for (ilayer=0 ; ilayer<=n_layers ; ilayer++) {
      gptr = hostgrad[ilayer] ;

/*
   Fully connected
*/

      if (ilayer == n_layers  ||  layer_type[ilayer] == TYPE_FC) {
         if (ilayer == 0) {
            in_n_height = img_rows ;
            in_n_width = img_cols ;
            in_n_depth = img_bands ;
            }
         else {
            in_n_height = height[ilayer-1] ;
            in_n_width = width[ilayer-1] ;
            in_n_depth = depth[ilayer-1] ;
            }
         n_prior = in_n_height * in_n_width * in_n_depth + 1 ;  // Number of weights per neuron, including bias
         if (ilayer == n_layers)
            n = n_classes ;  // Equals depth
         else
            n = nhid[ilayer] ;  // Equals depth
         for (idepth=0 ; idepth<n ; idepth++) {
            for (in_row=0 ; in_row<in_n_height ; in_row++) {
               for (in_col=0 ; in_col<in_n_width ; in_col++) {
                  for (in_slice=0 ; in_slice<in_n_depth ; in_slice++) {
                     // Compute location of this neuron's weight vector in host
                     isub = idepth * n_prior + (in_slice * in_n_height + in_row) * in_n_width + in_col ;
                     assert ( isub < n_weights ) ;
                     gptr[isub] += *fptr++ ;
                     } // For in_slice
                  } // For in_col
               } // For in_row

            // Bias
            isub = idepth * n_prior + n_prior - 1 ;
            assert ( isub < n_weights ) ;
            gptr[isub] += *fptr++ ;
            } // For idepth
         }

/*
   LOCAL
*/

      else if (layer_type[ilayer] == TYPE_LOCAL) {
         // For LOCAL layers, neuron layout in current layer is (height, width, depth).
         n = nhid[ilayer] ;
         ndepth = depth[ilayer] ;
         nheight = height[ilayer] ;
         nwidth = width[ilayer] ;
         in_n_height = 2 * hwV[ilayer] + 1 ;
         in_n_width = 2 * hwH[ilayer] + 1 ;
         if (ilayer == 0)
            in_n_depth = img_bands ;
         else
            in_n_depth = depth[ilayer-1] ;
         n_prior = in_n_height * in_n_width * in_n_depth + 1 ;  // Number of weights per neuron, including bias
         for (iheight=0 ; iheight<nheight ; iheight++) {  // nhid = ndepth * nheight * nwidth
            for (iwidth=0 ; iwidth<nwidth ; iwidth++) {   // We must reorder so depth changes fastest
               for (idepth=0 ; idepth<ndepth ; idepth++) {
                  for (in_row=0 ; in_row<in_n_height ; in_row++) {
                     for (in_col=0 ; in_col<in_n_width ; in_col++) {
                        for (in_slice=0 ; in_slice<in_n_depth ; in_slice++) {
                           // Compute location of this neuron's weight in host
                           isub = (idepth * nheight + iheight) * nwidth + iwidth ; // Neuron in this layer
                           isub = isub * n_prior + (in_slice * in_n_height + in_row) * in_n_width + in_col ;
                           assert ( isub < n_weights ) ;
                           gptr[isub] += *fptr++ ;
                           } // For in_slice
                        } // For in_col
                     } // For in_row
                  // Bias
                  isub = (idepth * nheight + iheight) * nwidth + iwidth ; // Neuron in this layer
                  isub = isub * n_prior + n_prior - 1 ;
                  assert ( isub < n_weights ) ;
                  gptr[isub] += *fptr++ ;
                  } // For idepth
               } // For iwidth
            } // For iheight
         }


/*
   CONV
*/

      else if (layer_type[ilayer] == TYPE_CONV) {
         nheight = height[ilayer] ;
         nwidth = width[ilayer] ;
         ndepth = depth[ilayer] ;
         in_n_height = 2 * hwV[ilayer] + 1 ;
         in_n_width = 2 * hwH[ilayer] + 1 ;
         if (ilayer == 0)
            in_n_depth = img_bands ;
         else
            in_n_depth = depth[ilayer-1] ;
         n_prior = in_n_height * in_n_width * in_n_depth + 1 ;  // Number of weights per neuron, including bias
         for (idepth=0 ; idepth<ndepth ; idepth++) {
            for (in_row=0 ; in_row<in_n_height ; in_row++) {
               for (in_col=0 ; in_col<in_n_width ; in_col++) {
                  for (in_slice=0 ; in_slice<in_n_depth ; in_slice++) {
                     // Compute location of this neuron's weight vector in host
                     isub = idepth * n_prior + (in_slice * in_n_height + in_row) * in_n_width + in_col ;
                     assert ( isub < n_weights ) ;
                     gptr[isub] += *fptr++ ;
                     } // For in_slice
                  } // For in_col
               } // For in_row
            //Bias
            isub = idepth * n_prior + n_prior - 1 ;
            assert ( isub < n_weights ) ;
            gptr[isub] += *fptr++ ;
            } // For idepth
         }

      } // For ilayer

   assert ( fptr == fdata + n_weights ) ;

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   CUDA_CLEANUP - Cleanup after CUDA processing

--------------------------------------------------------------------------------
*/

void cuda_cleanup ( int n_layers , int *layer_type )
{
   int i ;
   double sum ;
   char msg[256] ;

   MEMTEXT ( "CUDA cuda_cleanup starting" ) ;

   if (h_predictors != NULL) {
      cudaFree ( h_predictors ) ;
      h_predictors = NULL ;
      }

   if (h_class != NULL) {
      cudaFree ( h_class ) ;
      h_class = NULL ;
      }

   if (activations != NULL) {
      cudaFree ( activations ) ;
      activations = NULL ;
      }

   if (h_output != NULL) {
      cudaFree ( h_output ) ;
      h_output = NULL ;
      }

   for (i=0 ; i<n_layers ; i++) {
      if (h_poolmax_id[i] != NULL) {
         cudaFree ( h_poolmax_id[i] ) ;
         h_poolmax_id[i] = NULL ;
         }
      }

   if (weights != NULL) {
      cudaFree ( weights ) ;
      weights = NULL ;
      }

   if (grad != NULL) {
      cudaFree ( grad ) ;
      grad = NULL ;
      }

   if (h_convgrad_work != NULL) {
      cudaFree ( h_convgrad_work ) ;
      h_convgrad_work = NULL ;
      }

   if (h_this_delta != NULL) {
      cudaFree ( h_this_delta ) ;
      h_this_delta = NULL ;
      }

   if (h_prior_delta != NULL) {
      cudaFree ( h_prior_delta ) ;
      h_prior_delta = NULL ;
      }

   if (h_ll_out != NULL) {
      cudaFree ( h_ll_out ) ;
      h_ll_out = NULL ;
      }

   if (reduc_fdata != NULL) {
      FREE ( reduc_fdata ) ;
      reduc_fdata = NULL ;
      }

   if (fdata != NULL) {
      FREE ( fdata ) ;
      fdata = NULL ;
      }

   total_memory = 0.0 ;

   cudaDeviceReset () ;


/*
   Print CUDA timers
*/

   sum = 1.e-20 ;
   for (i=0 ; i<MAX_LAYERS ; i++) {
      sum += CudaTimers.act[i] ;
      sum += CudaTimers.delta[i] ;
      sum += CudaTimers.grad[i] ;
      }

   sum += CudaTimers.weights + CudaTimers.softmax + CudaTimers.ll + CudaTimers.movedelta + CudaTimers.fetchgrad ;

   cudalog ( "" ) ;
   cudalog ( "" ) ;
   cudalog ( "CUDA times in seconds: total, (percent), per launch" ) ;
   cudalog ( "" ) ;

   sprintf ( msg, "  Send weights =   %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.weights,
             100.0 * CudaTimers.weights / sum,
             0.001 * CudaTimers.weights / (CudaTimers.ncalls_weights + 1.e-20)) ;
   cudalog ( msg ) ;

   for (i=0 ; i<=n_layers ; i++) {
      if (i == n_layers)
         cudalog ( "  Output layer" ) ;
      else if (layer_type[i] == TYPE_FC) {
         sprintf ( msg, "  Layer %d is fully connected", i+1 ) ;
         cudalog ( msg ) ;
         }
      else if (layer_type[i] == TYPE_LOCAL) {
         sprintf ( msg, "  Layer %d is locally connected", i+1 ) ;
         cudalog ( msg ) ;
         }
      else if (layer_type[i] == TYPE_CONV) {
         sprintf ( msg, "  Layer %d is convolutional", i+1 ) ;
         cudalog ( msg ) ;
         }
      else if (layer_type[i] == TYPE_POOLAVG) {
         sprintf ( msg, "  Layer %d is pooled average", i+1 ) ;
         cudalog ( msg ) ;
         }
      else if (layer_type[i] == TYPE_POOLMAX) {
         sprintf ( msg, "  Layer %d is pooled max", i+1 ) ;
         cudalog ( msg ) ;
         }
      sprintf ( msg, "           act =   %8.3lf   (%5.1lf percent) %10.6lf per launch",
                0.001 * CudaTimers.act[i],
                100.0 * CudaTimers.act[i] / sum,
                0.001 * CudaTimers.act[i] / (CudaTimers.ncalls_act[i] + 1.e-20)) ;
      cudalog ( msg ) ;
      sprintf ( msg, "         delta =   %8.3lf   (%5.1lf percent) %10.6lf per launch",
                0.001 * CudaTimers.delta[i],
                100.0 * CudaTimers.delta[i] / sum,
                0.001 * CudaTimers.delta[i] / (CudaTimers.ncalls_delta[i] + 1.e-20)) ;
      cudalog ( msg ) ;
      sprintf ( msg, "          grad =   %8.3lf   (%5.1lf percent) %10.6lf per launch",
                0.001 * CudaTimers.grad[i],
                100.0 * CudaTimers.grad[i] / sum,
                0.001 * CudaTimers.grad[i] / (CudaTimers.ncalls_grad[i] + 1.e-20)) ;
      cudalog ( msg ) ;
      assert ( CudaTimers.grad[i] >= 0.0 ) ;
      assert ( CudaTimers.ncalls_grad[i] >= 0.0 ) ;
      assert ( (0.001 * CudaTimers.grad[i] / (CudaTimers.ncalls_grad[i] + 1.e-20)) >= 0.0 ) ;
      }

   sprintf ( msg, "  SoftMax =        %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.softmax,
             100.0 * CudaTimers.softmax / sum,
             0.001 * CudaTimers.softmax / (CudaTimers.ncalls_softmax + 1.e-20)) ;
   cudalog ( msg ) ;

   sprintf ( msg, "  Log likelihood = %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.ll,
             100.0 * CudaTimers.ll / sum,
             0.001 * CudaTimers.ll / (CudaTimers.ncalls_ll + 1.e-20)) ;
   cudalog ( msg ) ;

   sprintf ( msg, "  Move delta =     %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.movedelta,
             100.0 * CudaTimers.movedelta / sum,
             0.001 * CudaTimers.movedelta / (CudaTimers.ncalls_movedelta + 1.e-20)) ;
   cudalog ( msg ) ;

   sprintf ( msg, "  Fetch grad =     %8.3lf   (%5.1lf percent) %10.6lf per launch",
             0.001 * CudaTimers.fetchgrad,
             100.0 * CudaTimers.fetchgrad / sum,
             0.001 * CudaTimers.fetchgrad / (CudaTimers.ncalls_fetchgrad + 1.e-20)) ;
   cudalog ( msg ) ;

   MEMTEXT ( "CUDA cuda_cleanup ending" ) ;
}
