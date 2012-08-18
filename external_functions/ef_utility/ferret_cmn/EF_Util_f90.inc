!* EF_Util_f90.inc
!*
!* Jonathan Callahan
!* July 10th 1997
!*
!* This is the header file to be included by routines which
!* are part of the Ferret External Function library.
!*
!*       EXTERNAL ef_version 1.1: *jc* 9.98 Added       EXTERNAL ef_max_work_arrays
!* 
!*            1.3: *jc* 4.99 Changed "work_array_len" to "work_array_lo/hi" in EF_Util.h
!* 
!*                 *ac* 7.99 Added unspecified_int4
!*		   *sh* 8.99 Changed unspeficied_int4 to EF_unspecified_int4
!*				because of conflict with same-name parameter
!*				in ferret.parm
!*                 *ac* 9.99 Separate EF_Util.parm which is all that's needed 
!*                           in the utility fcns in fer/ef_utility and the 
!*                           EXTERNAL statements, to be included in EF code.
!*                 *ac* 5/06 Add ef_set_result_type
!*
!*            1.4: *kms*  3/12 Add Fortran interfacing C functions; add _6d functions
!*
!* Converted to f90 by Andrew Wittenberg, 8/14/2012

      INCLUDE 'ferret_cmn/EF_Util_f90.parm'

      EXTERNAL ef_bail_out
      EXTERNAL ef_get_arg_info
      EXTERNAL ef_get_arg_mem_subscripts
      EXTERNAL ef_get_arg_mem_subscripts_6d
      EXTERNAL ef_get_arg_ss_extremes
      EXTERNAL ef_get_arg_ss_extremes_6d
      EXTERNAL ef_get_arg_string
      EXTERNAL ef_get_arg_subscripts
      EXTERNAL ef_get_arg_subscripts_6d
      EXTERNAL ef_get_arg_type
      EXTERNAL ef_get_axis_dates
      EXTERNAL ef_get_axis_info
      EXTERNAL ef_get_axis_info_6d
      EXTERNAL ef_get_axis_modulo_len
      EXTERNAL ef_get_bad_flags
      EXTERNAL ef_get_box_hi_lim
      EXTERNAL ef_get_box_limits
      EXTERNAL ef_get_box_lo_lim
      EXTERNAL ef_get_box_size
      EXTERNAL ef_get_coordinates
      EXTERNAL ef_get_one_val
      EXTERNAL ef_get_res_mem_subscripts
      EXTERNAL ef_get_res_mem_subscripts_6d
      EXTERNAL ef_get_res_subscripts
      EXTERNAL ef_get_res_subscripts_6d
      EXTERNAL ef_get_result_type
      EXTERNAL ef_get_wrk_mem_subscripts
      EXTERNAL ef_get_wrk_mem_subscripts_6d
      EXTERNAL ef_set_arg_desc
      EXTERNAL ef_set_arg_name
      EXTERNAL ef_set_arg_type
      EXTERNAL ef_set_arg_unit
      EXTERNAL ef_set_axis_extend
      EXTERNAL ef_set_axis_influence
      EXTERNAL ef_set_axis_influence_6d
      EXTERNAL ef_set_axis_inheritance
      EXTERNAL ef_set_axis_inheritance_6d
      EXTERNAL ef_set_axis_limits
      EXTERNAL ef_set_axis_reduction
      EXTERNAL ef_set_axis_reduction_6d
      EXTERNAL ef_set_custom_axis
      EXTERNAL ef_set_desc
      EXTERNAL ef_set_has_vari_args
      EXTERNAL ef_set_num_args
      EXTERNAL ef_set_num_work_arrays
      EXTERNAL ef_set_piecemeal_ok
      EXTERNAL ef_set_piecemeal_ok_6d
      EXTERNAL ef_set_result_type
      EXTERNAL ef_set_work_array_dims
      EXTERNAL ef_set_work_array_dims_6d
      EXTERNAL ef_set_work_array_lens
      EXTERNAL ef_set_work_array_lens_6d
      EXTERNAL ef_version_test

