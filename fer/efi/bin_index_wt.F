*
* bin_index_wt.F
*
* Ansley Manke
* October 2002
*
* This function sorts the indicies, and gathers them into bins, along
* with their associated weights.  Returns the weights for each bin, 
* followed by missing-data after the last bin.
*
*
* In this subroutine we provide information about
* the function.  The user configurable information 
* consists of the following:
*
* descr              Text description of the function
*
* num_args           Required number of arguments
*
* axis_inheritance   Type of axis for the result
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )
* 
*
* For each argument we provide the following information:
*
* name               Text name for an argument
*
* unit               Text units for an argument
*
* desc               Text description of an argument
*
* axis_influence     Are this argument's axes the same as the result grid?
*                       ( YES, NO )
*
* axis_extend       How much does Ferret need to extend arg limits relative to result 
*


      SUBROUTINE bin_index_wt_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data, gathered into bins on the X axis')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, ABSTRACT, NORMAL, NORMAL, NORMAL)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'INDEX')
      CALL ef_set_arg_desc(id, arg, 'Bin indices: sorted, 1D on x axis')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'WT')
      CALL ef_set_arg_desc(id, arg, 'Weight for with each index')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'INDXMAX')
      CALL ef_set_arg_desc(id, arg, 'Maximum index value')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* lo_ss               lo subscript for an axis
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE bin_index_wt_result_limits(id)
      INCLUDE 'EF_Util.cmn'
      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo_l, my_hi_l
      REAL size

*
*     Use utility functions to get context information about the arguments.
*

      CALL ef_get_one_val(id, ARG3, size)

      my_lo_l = 1
      my_hi_l = INT(size + 1)

      CALL ef_set_axis_limits(id, X_AXIS, my_lo_l, my_hi_l)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


* In this subroutine we compute the result
*
      SUBROUTINE bin_index_wt_compute(id, arg_1, arg_2, arg_3,result)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .           mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .           mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER id
      INTEGER ndx, m_indx
      INTEGER m, nsrt
      INTEGER i, j, k, l
      INTEGER i1, j1, k1, l1

*  Get subscript ranges

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


      j1 = arg_lo_ss(Y_AXIS, ARG1)
      k1 = arg_lo_ss(Z_AXIS, ARG1)
      l1 = arg_lo_ss(T_AXIS, ARG1)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)

      DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
          result(i,j,k,l) = 0.0
  100 CONTINUE

      ndx = res_lo_ss(X_AXIS)
      DO 200 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
          IF (INT(arg_1(i1,j1,k1,l1)) .EQ. ndx) THEN
              result(ndx,j,k,l) = result(ndx,j,k,l) + arg_2(i1,j1,k1,l1)
          ELSEIF (INT(arg_1(i1,j1,k1,l1)) .GT. 0) THEN
              ndx = INT(arg_1(i1,j1,k1,l1))
              result(ndx,j,k,l) = arg_2(i1,j1,k1,l1)
          ENDIF
  200 CONTINUE

      DO 300 i = ndx+1, res_hi_ss(X_AXIS)
         result(i,j,k,l) = bad_flag_result
  300 CONTINUE
      
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END
