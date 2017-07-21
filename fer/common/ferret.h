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
#define NFERDIMSP1 7

#define TTOUT_LUN 6

#define FRTN_CONTROL  0    /* 1 in FORTRAN */
#define FRTN_ACTION   2    /* 3 in FORTRAN */
#define FRTN_IDATA1   5    /* 6 in FORTRAN */
#define FRTN_IDATA2   6    /* 7 in FORTRAN */

#define FCTRL_IN_FERRET    2

#define FACTN_MEM_RECONFIGURE  1
#define FACTN_EXIT             2

#ifndef DFTYPE
#ifdef double_p
#define DFTYPE double
#else
#define DFTYPE float
#endif
#endif

/* Prototypes for C functions */
FILE *executableOutput(char *exeArgv[], pid_t *childPidPtr, char errMsg[]);
void ferret_dispatch_c(char *init_command, smPtr sBuffer);
int  getJavaVersion(char javaExeName[], char errMsg[]);
int  runThreddsBrowser(char datasetName[], char errWarn[]);
void set_batch_graphics( char *outfile, int *pngonly );
void set_secure(void);
void set_server(void);

void FORTRAN(assign_modemeta)(void);
void FORTRAN(c_dncase)(char **in_ptr, char **out_ptr);
void FORTRAN(c_strcat)(char **in_ptr1, char **in_ptr2, char **out_ptr);
void FORTRAN(c_strcmp)(char **in_ptr1, char **in_ptr2, int *out_ptr);
void FORTRAN(c_strfloat)(char **in_ptr, DFTYPE *out_ptr, DFTYPE *bad_ptr);
void FORTRAN(c_strindex)(char **in_ptr1, char **in_ptr2, int *out_ptr);
void FORTRAN(c_strlen)(char **in_ptr, int *out_ptr);
void FORTRAN(c_strrindex)(char **in_ptr1, char **in_ptr2, int *out_ptr);
void FORTRAN(c_substr)(char **in_ptr, int *offset, int *length, char **out_ptr);
void FORTRAN(c_upcase)(char **in_ptr, char **out_ptr);
void FORTRAN(cache_full_array)(double *array, int *alen, double **pointer_val);
int  FORTRAN(compare_c_strings)(char **ptr_1, char **ptr_2);
void FORTRAN(copy_c_string)(char **in_ptr, char **out_ptr);
void FORTRAN(create_utf8_str)(const int *codepoint, char *utf8str, int *utf8strlen);
void FORTRAN(date_decode)(char *strdate, DFTYPE *rdum);
void FORTRAN(dynmem_free_ptr_array)(DFTYPE ***mr_ptrs_ptr);
void FORTRAN(dynmem_make_ptr_array)(int* n, DFTYPE ***mr_ptrs_ptr, int* status);
void FORTRAN(dynmem_pass_1_ptr)(int* iarg, DFTYPE* arg_ptr, DFTYPE ***mr_ptrs_ptr);
void FORTRAN(free_c_pointer)(char ***fer_ptr);
void FORTRAN(free_c_string_array)(char ***fer_ptr, int *length);
void FORTRAN(free_cached_full_array)(double **pointer_val);
void FORTRAN(free_dyn_mem)(double *mvar);
void FORTRAN(get_c_pointer)(char** mr_ptr, char** mr_ptr_val);
void FORTRAN(get_c_string)(char **ptr_ptr, char *outstring, int *maxlen);
int  FORTRAN(get_c_string_len)(char **ptr_ptr);
int  FORTRAN(get_max_c_string_len)(char ***fer_ptr, int *nstr);
void FORTRAN(get_mr_mem)(double *index, int *alen, int *status);
void FORTRAN(get_offset_c_string)(char ***fer_ptr, int *offset, char *outstring, int *maxlen);
int  FORTRAN(get_offset_c_string_len)(char ***fer_ptr, int *offset);
void FORTRAN(get_sys_cmnd)(char ***fer_ptr, int *nlines, char *cmd, int *stat);
void FORTRAN(get_ws_mem)(int *index, int *alen, int *status);
void FORTRAN(init_c_string_array)(int *length, char **mr_blk1, char ***fer_ptr);
int  FORTRAN(is_secure)(void);
int  FORTRAN(is_server)(void);
void FORTRAN(linux_perror)(char *string);
void FORTRAN(nullify_mr)(int *mr);
void FORTRAN(nullify_ws)(int *ws);
void FORTRAN(put_frame)(int *ws_id, char *filename, char *errstr, char *format, int *status);
void FORTRAN(put_frame_batch)(int *ws_id, char *filename, char *format, int *transp, 
                              DFTYPE *red, DFTYPE *green, DFTYPE *blue, char *errmsg, int *status);
void FORTRAN(recover_cached_2d_array_point)(double *array, int *dim1, int *dim2, int *i, int *j, double **pointer_val);
void FORTRAN(recover_cached_full_array)(double *array, int *alen, double **pointer_val);
int  FORTRAN(replaceable_bad_flags)(DFTYPE *bad1, DFTYPE *bad2);
void FORTRAN(replace_bad_data_sub)(DFTYPE *old_bad, DFTYPE *src, int *size, DFTYPE *new_bad);
int  FORTRAN(run_thredds_browser)(char dataset_name[], char err_warn_msg[], int max_len_data_set, int max_len_err_warn_msg);
void FORTRAN(save_c_string)(char *string, int *inlen, char ***fer_ptr, int *offset, int *stat);
void FORTRAN(save_metafile_name)(char *outfile, int *length);
void FORTRAN(set_batch_graphics)(char *outfile, int *batmode);
void FORTRAN(set_nan)(DFTYPE *val);
void FORTRAN(set_null_c_string)(char **out_ptr);
void FORTRAN(set_null_c_string_array)(char ***fer_ptr, int *nstr);
int  FORTRAN(sizeof_pointer)(void);
void FORTRAN(store_mr_ptr)(double *index, int *alen, double *pointer);
void FORTRAN(store_nul_mr_ptr)(int *mr, double *nul_ptr);
void FORTRAN(store_nul_ws_ptr)(int *ws, double *nul_ptr);
void FORTRAN(store_ws_ptr)(int *index, int *alen, double *pointer);
void FORTRAN(text_to_utf8)(const char *text, const int *textlen, char *utf8str, int *utf8strlen);
DFTYPE FORTRAN(time_decode)(char *strtime);
void FORTRAN(us2i_str_cmp)(char *str1, char *str2, int *ival);
void FORTRAN(us2i_compare_string_list)(char* compare_string, int *str_seq);
int  FORTRAN(write_dods)(char*filename, int* slen, int *clobber, int *swap, int *length, float *data);
int  FORTRAN(write_dods_double)(char*filename, int* slen, int *clobber, int *swap, int *length, double *data);
void FORTRAN(xfer_c_ptrs)(char ***src_ptr, int *src_del, int *src_offset,
		          char ***dst_ptr, int *dst_del, int *dst_offset, int *nptr);

/* Prototypes for Fortran functions called by C functions */
void FORTRAN(ctrlc_ast)(void);
DFTYPE FORTRAN(days_from_day0)(double *days1900, int* iyr, int* imon, int* iday, DFTYPE* rdum, int* status);
void FORTRAN(ferret_dispatch)(char *init_command, int *rtn_flags, int *nflags, char *rtn_chars, int *nchars, int *nerrlines);
void FORTRAN(finalize_ferret)(void);
void FORTRAN(init_journal)(int *status);
void FORTRAN(init_memory)(DFTYPE *vmem_size_arg);
void FORTRAN(initialize_ferret)(void);
void FORTRAN(no_journal)(void);
void FORTRAN(save_frame_name)(char *outfile,  int *length);
void FORTRAN(save_scriptfile_name)(char *name, int *clen, int *status);
void FORTRAN(set_ctrl_c)(void (*func)(void));
void FORTRAN(proclaim_c)(int *ttoutLun, char *leader, int *quiet);
void FORTRAN(get_scriptfile_name)(char *name, int *ipath, int name_size);
void FORTRAN(turnoff_verify)(int *status);
void FORTRAN(version_only)(void);

#endif /* _FERRET_H */
