#ifndef _FERRET_H 
#define _FERRET_H

#include <sys/types.h>
#include <stdio.h>

#include "ferret_shared_buffer.h"

/* Easier way of handling FORTRAN calls with underscore/no underscore */
#ifndef FORTRAN
#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif
#endif

#define NFERDIMS 6

#define TTOUT_LUN 6

#define FRTN_CONTROL  0    /* 1 in FORTRAN */
#define FRTN_ACTION   2    /* 3 in FORTRAN */
#define FRTN_IDATA1   5    /* 6 in FORTRAN */
#define FRTN_IDATA2   6    /* 7 in FORTRAN */

#define FCTRL_IN_FERRET    2

#define FACTN_MEM_RECONFIGURE  1
#define FACTN_EXIT             2

#ifdef double_p
#define DFTYPE double
#else
#define DFTYPE float
#endif

/* Prototypes for C functions */
FILE *executableOutput(char *exeArgv[], pid_t *childPidPtr, char errMsg[]);
void ferret_dispatch_c( char *init_command, smPtr sBuffer );
int  getJavaVersion(char javaExeName[], char errMsg[]);
int  runThreddsBrowser(char datasetName[], char errWarn[]);
void set_batch_graphics( char *outfile, int *pngonly );
void set_secure( void );
void set_server( void );

void FORTRAN(create_utf8_str)(const int *codepoint, char *utf8str, int *utf8strlen);
void FORTRAN(dynmem_free_ptr_array)( long* mr_ptrs_val );
void FORTRAN(dynmem_make_ptr_array)( int* n, long* mr_ptrs_val, int* status );
void FORTRAN(dynmem_pass_1_ptr)( int* iarg, DFTYPE* arg_ptr, long* mr_ptrs_val );
void FORTRAN(free_dyn_mem) ( double *mvar );
void FORTRAN(get_mr_mem)( double *index, int *alen, int *status );
int  FORTRAN(is_secure)( void );
int  FORTRAN(is_server)( void );
int  FORTRAN(run_thredds_browser)(char dataset_name[], char err_warn_msg[], 
                                  int max_len_data_set, int max_len_err_warn_msg);
void FORTRAN(text_to_utf8)(const char *text, const int *textlen, char *utf8str, int *utf8strlen);

/* Prototypes for Fortran functions called by C functions */
void FORTRAN(ferret_dispatch)( char *init_command, int *rtn_flags, int *nflags, 
		               char *rtn_chars, int *nchars, int *nerrlines );


#endif /* _FERRET_H */

