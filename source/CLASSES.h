/******************************************************************************/
/*                                                                            */
/*  CLASSES.H - Many class and struct definitions are here                    */
/*                                                                            */
/******************************************************************************/

// Note to readers... This includes much information not directly related to
// material in the book.  I left a lot of such material here just so readers
// could get some vague ideas of what I do in the program.
// Many readers would be best off ignoring this CLASSES.H file entirely!
// The only reason it's here at all is to serve as a concise reference
// to Model variables and parameters.

/*
   Fundamental structure of a model
*/

typedef struct {
   int n_layers ;               // Number of layers, not including final
   int layer_type[MAX_LAYERS] ; // Each entry (input to final) is TYPE_? in CONST.H
   int depth[MAX_LAYERS] ;      // Number of hidden neurons if fully connected, else number of slices
   int HalfWidH[MAX_LAYERS] ;   // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ;   // And vertical
   int padH[MAX_LAYERS] ;       // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ;       // And vertical
   int strideH[MAX_LAYERS] ;    // Horizontal stride
   int strideV[MAX_LAYERS] ;    // And vertical
   int PoolWidH[MAX_LAYERS] ;   // Horizontal pooling width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ;   // And vertical
} ARCHITECTURE ;


/*
   Training parameters
*/

typedef struct {
   int max_batch ;       // Divide the training set into subsets for CUDA timeouts; this is max size of a subset
   int max_hid_grad ;    // Maximum number of hidden neuron per CUDA launch; prevents timeout error and lowers memory use
   int max_mem_grad ;    // Maximum CONV working memory (MB) per CUDA launch; prevents timeout error and lowers memory use
   int anneal_iters ;    // Supervised annealing iters
   double anneal_rng ;   // Starting range for annealing
   int maxits ;          // Max iterations for training supervised section
   double tol ;          // Convergence tolerance for training supervised section
   double wpen ;         // Weight penalty (should be very small)
   // These are set in READ_SERIES.CPP and copied to model during training
   int class_type ;      // 1=split at zero; 2=split at median; 3=split at .33 and .67 quantiles; READ_SERIES.CPP sets, MODEL.CPP uses
   double median ;
   double quantile_33 ;
   double quantile_67 ;
} TRAIN_PARAMS ;


/*
   CUDA timers
*/

typedef struct {
   int ncalls_weights ;
   int weights ;
   int ncalls_act[MAX_LAYERS+1] ;
   int act[MAX_LAYERS+1] ;
   int ncalls_softmax ;
   int softmax ;
   int ncalls_ll ;
   int ll ;
   int ncalls_delta[MAX_LAYERS+1] ;
   int delta[MAX_LAYERS+1] ;
   int ncalls_grad[MAX_LAYERS+1] ;
   int grad[MAX_LAYERS+1] ;
   int ncalls_movedelta ;
   int movedelta ;
   int ncalls_fetchgrad ;
   int fetchgrad ;
} CUDA_TIMERS ;


/*
--------------------------------------------------------------------------------

   Model - This is the model

--------------------------------------------------------------------------------
*/

class Model {

public:

   Model ( ARCHITECTURE *arc , int nprd , int ncls ) ;
   ~Model () ;
   int train ( int istart , int istop , int print ) ;
   void print_architecture () ;
   void print_train_params () ;
   void print_weights () ;
   void find_final_weights ( int istart , int istop ) ;
   double trial_error ( int istart , int istop ) ;
   double grad ( int istart , int istop ) ;

   int ok ;                     // Did memory allocation go okay?
   int ok_to_test ;             // Set when model is trained, reset if user changes Architecture
   double crit ;                // Trained performance criterion
   double penalty ;             // Penalty associated with trained weights

private:
   // The next four declarations are defined in MOD_NO_THR.CPP
   // These 'No threading' routines are not normally used; they can be forced at the end of MODEL.CPP for testing.
   // By putting them in the model we can use private variables in them, simplifying the calling parameter list.
   void activity_local_no_thr ( int ilayer , double *input ) ;
   void activity_conv_no_thr ( int ilayer , double *input ) ;
   void activity_fc_no_thr ( int ilayer , double *input , int nonlin ) ;
   void activity_pool_no_thr ( int ilayer , double *input ) ;
   void trial_no_thr ( double *input ) ;
   double model_cuda ( int find_grad , int jstart , int jstop ) ;
   double trial_error_cuda ( int istart , int istop ) ;
   double trial_error_no_thr ( int istart , int istop ) ;
   double trial_error_thr ( int istart , int istop ) ;
   double grad_no_thr ( int istart , int istop ) ;
   double grad_thr ( int istart , int istop ) ;
   double grad_cuda ( int istart , int istop ) ;
   void grad_no_thr_FC ( int icase , int ilayer ) ;
   void grad_no_thr_LOCAL ( int icase , int ilayer ) ;
   void grad_no_thr_CONV ( int icase , int ilayer ) ;
   void grad_no_thr_POOL ( int ilayer ) ;
   void compute_nonpooled_delta ( int ilayer ) ;
   void compute_pooled_delta ( int ilayer ) ;
   double gamma ( double *g , double *grad ) ;
   void find_new_dir ( double gam , double *g , double *h , double *grad ) ;
   void check_grad ( int istart , int istop ) ;
   void step_out ( double step , double *direc , double *base ) ;
   void update_dir ( double step , double *direc ) ;
   void negate_dir ( double *direc ) ;
   double direcmin ( int istart , int istop , double start_err , int itmax , double eps , double tol ,
                     double *base , double *direc ) ;
   double conjgrad ( int istart , int istop , int maxits , double reltol , double errtol ) ;
   void confuse ( int train_vs_test , int istart , int istop , int *summary_confusion , int print ) ;

   int n_pred ;                 // Number of predictors present (input grid size)
   int n_classes ;              // Number of classes
   int n_layers ;               // Number of layers, not including final
   int layer_type[MAX_LAYERS] ; // Each entry is TYPE_? in CONST.H
   int height[MAX_LAYERS] ;     // Number of neurons vertically in a slice of this layer, 1 if fully connected
   int width[MAX_LAYERS] ;      // Ditto horizontal
   int depth[MAX_LAYERS] ;      // Number of hidden neurons if fully connected, else number of slices in this layer
   int nhid[MAX_LAYERS] ;       // Total number of neurons in this layer = height times width times depth
   int HalfWidH[MAX_LAYERS] ;   // Horizontal half width looking back to prior layer
   int HalfWidV[MAX_LAYERS] ;   // And vertical
   int padH[MAX_LAYERS] ;       // Horizontal padding, should not exceed half width
   int padV[MAX_LAYERS] ;       // And vertical
   int strideH[MAX_LAYERS] ;    // Horizontal stride
   int strideV[MAX_LAYERS] ;    // And vertical
   int PoolWidH[MAX_LAYERS] ;   // Horizontal pooling width looking back to prior layer
   int PoolWidV[MAX_LAYERS] ;   // And vertical
   int n_prior_weights[MAX_LAYERS+1] ; // N of inputs per neuron (including bias) from prior layer = prior depth * (2*HalfWidH+1) * (2*HalfWidV+1) + 1
                                // A CONV layer has this many weights per layer (slice); a LOCAL layer has this times its nhid
   int n_hid_weights ;          // Total number of all hidden weights; includes bias; does not include final layer
   int n_all_weights ;          // As above, but also includes final layer weights (including bias)
   int max_any_layer ;          // Max number of neurons in any layer, including input and output
   double *weights ;            // All 'n_all_weights' weights, including final weights, are here, in order by layer
   double *layer_weights[MAX_LAYERS+1] ; // Pointers to each layer's weights in 'weight' vector
   double *best_wts ;           // Scratch vector for simulated annealing
   double *center_wts ;         // Ditto
   double *gradient ;           // 'n_all_weights' gradient, aligned with weights
   double *layer_gradient[MAX_LAYERS+1] ; // Pointers to each layer's gradient in 'gradient' vector
   double *activity[MAX_LAYERS] ; // Activity vector for each layer
   double *this_delta ;         // Scratch vector for gradient computation
   double *prior_delta ;        // Ditto
   double output[MAX_CLASSES] ; // SoftMax activation for each class
   double *conf_scratch ;       // Scratch vector for CONFUSE.CPP
   int *poolmax_id[MAX_LAYERS] ;// Used only for POOLMAX layer; saves from forward pass ID of max input for backprop pass
   int *confusion ;             // Confusion matrix
   double *pred ;               // All case outputs; used in CONFUSE.CPP
   double *thresh ;             // Thresholds for tail classification; used in CONFUSE.CPP (alloc in MODEL.CPP)
   // These are allocated and freed in Model::train() (MOD_TRAIN.CPP) to provide multiple threads with private work areas
   double *thr_output ;
   double *thr_this_delta ;
   double *thr_prior_delta ;
   double *thr_activity[MAX_THREADS][MAX_LAYERS] ;
   int *thr_poolmax_id[MAX_THREADS][MAX_LAYERS] ;
   double *thr_gradient[MAX_THREADS] ;
   double *thr_layer_gradient[MAX_THREADS][MAX_LAYERS+1] ;
   // These preserve thresholds for testing after training
   int class_type ;             // 1=split zt zero; 2=split at median; 3=split at .33 and .67 quantiles
   double median ;
   double quantile_33 ;
   double quantile_67 ;
} ;
