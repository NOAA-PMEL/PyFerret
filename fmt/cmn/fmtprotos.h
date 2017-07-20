#ifndef _FMTPROTOS_H_
#define _FMTPROTOS_H_

/* Better if these were defined in only one include file, but .... */
#ifndef FORTRAN
#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif
#endif

#ifndef DFTYPE
#ifdef double_p
#define DFTYPE double
#else
#define DFTYPE float
#endif
#endif

int  high_ver_name(char *name, char *path);
int  string_array_hash(unsigned char *key, unsigned long int length, unsigned long int initval, int range);
void tm_blockify_ferret_strings(char **mr_blk1, char *pblock, int bufsiz, int outstrlen);
void tm_unblockify_ferret_strings(char **mr_blk1, char *pblock, int bufsiz, int filestrlen);

int FORTRAN(anal_file)(char *fname, char *recptr, char *delims, int *skip, int *maxrec, 
                       int* reclen, int* nfields, int field_type[], int *max_fields);
void FORTRAN(cd_read_scale)(int *cdfid, int *varid, int *dims, DFTYPE *offset, DFTYPE *scale, DFTYPE* bad,
                            int *tmp_start, int *tmp_count, int *tmp_stride, int *tmp_imap, 
                            void *dat, int *permuted, int *strided, int *already_scaled, int *cdfstat, int *status);
void FORTRAN(cd_read_sub)(int *cdfid, int *varid, int *dims, 
                          int *tmp_start, int *tmp_count, int *tmp_stride, int *tmp_imap, 
                          char **dat, int *permuted, int *strided, int *cdfstat);
void FORTRAN(cd_write_att_dp_sub)(int *cdfid, int *varid, char* attname, int *attype, int *nval, void *val, int *status);
void FORTRAN(cd_write_att_sub)(int *cdfid, int *varid, char* attname, int *attype, int *nval, void *val, int *status);
void FORTRAN(cd_write_var_sub)(int *cdfid, int *varid, int *vartyp, int *dims, 
                               int *tmp_start, int *tmp_count, int *strdim, void *dat, int *cdfstat);
void FORTRAN(deleted_list_clear)(void **deleted_list_header);
void FORTRAN(deleted_list_get_del)(void **deleted_list_header, int *result_array, int *result_array_size, int *num_indices);
void FORTRAN(deleted_list_get_undel)(void **deleted_list_header, int *result_array, int *result_array_size, int *num_indices);
void FORTRAN(deleted_list_init)(void **deleted_list_header, int *int_array, int *int_array_size, int *deleted_value);
void FORTRAN(deleted_list_modify)(void **deleted_list_header, int *index, int *new_value);
void FORTRAN(decode_file_jacket)(char* fname, char *recptr, char *delims, int *skip, int* maxrec, int* reclen, 
                                 int* nfields, int field_type[], int* nrec, int mrlist[], long* mr_ptrs_val, 
                                 DFTYPE mr_bad_flags[], char ***mr_c_ptr, int* status);
void FORTRAN(str_case_blind_compare_sub)(char *test_name, int *len_test, char *model_name, int *len_model, int *result);
void FORTRAN(str_dncase_sub)(char *out_string, int *out_len, char *in_string, int *in_len);
void FORTRAN(string_array_clear)(void **string_array_header);
void FORTRAN(string_array_find)(void **string_array_header, char *test_string, int *test_len, 
                                int *result_array, int *result_array_size, int *num_indices);
void FORTRAN(string_array_find_caseblind)(void **string_array_header, char *test_string, int *test_len, 
                                          int *result_array, int *result_array_size, int *num_indices);
void FORTRAN(string_array_find_exact)(void **string_array_header, char *test_string, int *test_len, 
                                      int *result_array, int *result_array_size, int *num_indices);
void FORTRAN(string_array_find_quoted)(void **string_array_header, char *test_string, int *test_len, 
                                       int *result_array, int *result_array_size, int *num_indices);
void FORTRAN(string_array_get_strlen)(void **string_array_header, int *index, int *true_strlen);
void FORTRAN(string_array_get_strlen1)(void **string_array_header, int *index, int *true_strlen);
void FORTRAN(string_array_init)(void **string_array_header, int *array_size, int *string_size, char *string_array);
void FORTRAN(string_array_modify)(void **string_array_header, int *index, char *new_string, int *new_string_size);
void FORTRAN(string_array_modify_upcase)(void **string_array_header, int *index, char *new_string, int *new_string_size);
void FORTRAN(str_upcase_sub)(char *out_string, int *out_len, char *in_string, int* in_len);
void FORTRAN(switch_nan)(DFTYPE *bad, DFTYPE *missing, DFTYPE *bad_val);
int  FORTRAN(tm_break_fmt_date_c)(char *date, int *year, int *month, int *day, int *hour, int *minute, DFTYPE *second);
void FORTRAN(tm_c_rename)(char *oldname, char *newname, int *status);
char *FORTRAN(tm_c_ver_name)(char *name, char *next_name, char *path);
int  FORTRAN(tm_check_inf)(DFTYPE *src);
int  FORTRAN(tm_check_nan)(DFTYPE *src);
void FORTRAN(tm_ep_time_convrt)(int *epjday, int *epmsec, int *mon, int *day, int *yr, int *hour, int *min, DFTYPE *sec);
int  FORTRAN(tm_ftoc_readline)(char *prompt, char *buff);
void FORTRAN(tm_get_strlen)(int *len_str, int *whole_len, char *in_string);
void FORTRAN(tm_make_relative_ver)(char *curr_ver, char *fname, char *path, int *real_ver);
void FORTRAN(tm_match_captial_name)(char *test_name, char *model_name, int *len_str, int *result);
void FORTRAN(tm_number_sub)(char *string, int *result);
void FORTRAN(tm_set_free_event)(int *n);
double FORTRAN(tm_world_recur)(int *isubscript, int *iaxis, int *where_in_box, int *max_lines, 
                               double line_mem[], int line_parent[], int line_class[], int line_dim[], 
                               double line_start[], double line_delta[], int line_subsc1[], 
                               int line_modulo[], double line_modulo_len[], int line_regular[]);
int  FORTRAN(url_encode)(char *str, char *outstr, int *outlen);

/* defined in NCF_Util.c */
int  FORTRAN(ncf_inq_ds)( int *, int *, int *, int *, int *);
int  FORTRAN(ncf_inq_ds_dims)( int *, int *, char *, int *, int *);
int  FORTRAN(ncf_inq_var) (int *, int *, char *, int *, int *, int *, int *, int *, int *, int * );

int  FORTRAN(ncf_inq_var_att)( int *, int *, int *, char *, int *, int *, int *, int *);

int  FORTRAN(ncf_get_dsnum)( char * );
int  FORTRAN(ncf_get_dsname)( int *, char *);
int  FORTRAN(ncf_get_dim_id)( int *, char *);

int  FORTRAN(ncf_get_var_name)( int *, int *, char *, int *);
int  FORTRAN(ncf_get_var_id)( int *, int*, char *);
int  FORTRAN(ncf_get_var_id_case)( int *, int*, char *);
int  FORTRAN(ncf_get_var_axflag)( int *, int *, int *, int *);
int  FORTRAN(ncf_get_var_attr_name) (int *, int *, int *, int *, char*);
int  FORTRAN(ncf_get_var_attr_id) (int *, int *, char* , int*);
int  FORTRAN(ncf_get_var_attr_id_case) (int *, int *, char* , int*);
int  FORTRAN(ncf_get_var_attr) (int *, int *, char* , char* , int *, double *);
int  FORTRAN(ncf_get_attr_from_id) (int *, int *, int * , int *, double* );

int  FORTRAN(ncf_get_var_outflag) (int *, int *, int *);
int  FORTRAN(ncf_get_var_outtype) (int *, int *, int *);
int  FORTRAN(ncf_get_var_type) (int *, int *, int *);
int  FORTRAN(ncf_get_var_uvflag) (int *, int *, int *);

int  FORTRAN(ncf_init_uvar_dset)( int *);
int  FORTRAN(ncf_init_uax_dset)( int *);
void FORTRAN(ncf_datasets_list_clear)(void);
int  FORTRAN(ncf_add_dset)( int *, int *, char *, char *);
int  FORTRAN(ncf_init_other_dset)( int *, char *, char *);
int  FORTRAN(ncf_delete_dset)( int *);
int  FORTRAN(ncf_delete_var_att)( int *, int *, char *);
int  FORTRAN(ncf_delete_var)( int *, char *);

int  FORTRAN(ncf_add_var)( int *, int *, int *, int *, char *, char *, char *, double *);
int  FORTRAN(ncf_add_coord_var)( int *, int *, int *, int *, char *, char *, double *);

int  FORTRAN(ncf_add_var_num_att)( int *, int *, char *, int *, int *, int *, DFTYPE *);
int  FORTRAN(ncf_add_var_num_att_dp)( int *, int *, char *, int *, int *, int *, double *);
int  FORTRAN(ncf_add_var_str_att)( int *, int *, char *, int *, int *, int *, char *);

int  FORTRAN(ncf_rename_var)( int *, int *, char *);
int  FORTRAN(ncf_rename_dim)( int *, int *, char *);

int  FORTRAN(ncf_repl_var_att)( int *, int *, char *, int *, int *, DFTYPE *, char *);
int  FORTRAN(ncf_repl_var_att_dp)( int *, int *, char *, int *, int *, double *, char *);
int  FORTRAN(ncf_set_att_flag)( int *, int *, char *, int *);
int  FORTRAN(ncf_set_var_out_flag)( int *, int *, int *);
int  FORTRAN(ncf_set_var_outtype)( int *, int *, int *);
int  FORTRAN(ncf_set_axdir)(int *, int *, int *);
int  FORTRAN(ncf_transfer_att)(int *, int *, int *, int *, int *);

int  FORTRAN(ncf_init_agg_dset)( int *, char *);
int  FORTRAN(ncf_add_agg_member)( int *, int *, int *);
int  FORTRAN(ncf_get_agg_count)( int *, int *);
int  FORTRAN(ncf_get_agg_member)( int *, int *, int *);
int  FORTRAN(ncf_get_agg_var_info)( int *, int *, int *, int *, int *, int *, int *, int *);
int  FORTRAN(ncf_put_agg_memb_grid)( int *, int *, int *, int *);

/* uvar grid management functions */
int  FORTRAN(ncf_free_uvar_grid_list)( int *, int *);
int  FORTRAN(ncf_set_uvar_grid)( int *, int *, int *, int *, int *);
int  FORTRAN(ncf_get_uvar_grid)( int *, int *, int *, int *, int *);
int  FORTRAN(ncf_set_uvar_aux_info)( int *, int *, int *, int *, int *);
int  FORTRAN(ncf_get_uvar_aux_info)( int *, int *, int *, int *, int *);
int  FORTRAN(ncf_get_uvar_grid_list_len)( int *, int *, int *);
int  FORTRAN(ncf_delete_uvar_grid)( int *, int *, int *);
int  FORTRAN(ncf_next_uvar_grid_in_list)( int *, int *, int *, int *);

/* Fortran functions called by C functions */
int  FORTRAN(free_time)(void);
int  FORTRAN(tm_its_subspan_modulo_int)(int *axis);
void FORTRAN(tm_ww_axlims)(int *axis, double *lo, double *hi);
double FORTRAN(tm_modulo_axlen)(int *axis);

#endif
