* 
*  expndi_id_by_z_counts.F
* 
*  Ansley Manke (from expndi_by_z)
*  8/24/2011
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
* 
* Create a variable with profile ID, where argument 1 is the length
* of the variable to be put into an XZ structure from it with the 
* Z length specified by column 2.  This creates an "ID" variable to
* go along with the XZ variable defined by expndi_by_z_counts.
 
*  In this subroutine we provide information about
*  the function.  The user configurable information 
*  consists of the following:
* 
*  descr              Text description of the function
* 
*  num_args           Required number of arguments
* 
*  axis_inheritance   Type of axis for the result
*                        ( CUSTOM, IMPLIED_BY_Z_ARGS, NORMAL, ABSTRACT )
*                        CUSTOM          - user defined axis
*                        IMPLIED_BY_Z_ARGS - same axis as the incoming argument
*                        NORMAL          - the result is normal to this axis
*                        ABSTRACT        - an axis which only has index values
* 
*  piecemeal_ok       For memory optimization:
*                        axes where calculation may be performed piecemeal
*                        ( YES, NO )
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


      SUBROUTINE expndi_id_by_z_counts_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .  'Create 2D list, ID by profile-length on an XZ grid.' )

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .     IMPLIED_BY_ARGS, ABSTRACT, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'counts')
      CALL ef_set_arg_desc(id, arg,
     .          'count of values in each profile')
      CALL ef_set_axis_influence(id, arg, NO, YES, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'max profile len')
      CALL ef_set_arg_desc(id, arg,
     .          'K size of result (constant)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

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

      SUBROUTINE expndi_id_by_z_counts_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      REAL size
      integer arg, nx, nz
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the arguments.
*
* output X axis size
* output Z axis size

* Length of x

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      arg = 1
      nx = arg_hi_ss(X_AXIS,arg) - arg_lo_ss(X_AXIS,arg) + 1
      CALL ef_set_axis_limits(id, X_AXIS, 1, nx)

* Length of z 
      CALL ef_get_one_val(id, ARG2, size)
      nz = INT(size)

      CALL ef_set_axis_limits(id, X_AXIS, 1, nx)
      CALL ef_set_axis_limits(id, Z_AXIS, 1, nz)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



* 
*  In this subroutine we compute the result
* 
      SUBROUTINE expndi_id_by_z_counts_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy,
     .	   mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .     mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1
      INTEGER iout, kout, isum
      REAL size, xlen, zlen

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* Length of x*z 

      xlen = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1
      zlen = res_hi_ss(Z_AXIS) - res_lo_ss(Z_AXIS) + 1

* arg 1 must not depend on z
      IF ( (arg_lo_ss(Z_AXIS,ARG1) .NE. arg_hi_ss(Z_AXIS,ARG1)) ) CALL
     .           EF_BAIL_OUT(id, 'Z size of arg 1 must be 1')

* pre-fill the entire output result with missing value flags
*  (since length of each profile may have different number of pts)
      DO 60 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         DO 50 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            DO 40 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               DO 30 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
                  result(i,j,k,l) = bad_flag_result
 30            CONTINUE
 40         CONTINUE
 50      CONTINUE
 60   CONTINUE

* fill in the points as specified by the counts
      k1 = arg_lo_ss(Z_AXIS,ARG1)

      l1 = arg_lo_ss(T_AXIS,ARG1)
      DO 400 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            i1 = 1
            DO 200 iout = 1, NINT(xlen)
               DO 100 kout = 1, NINT(arg_1(iout,j1,k1,l1))
                  result(iout,j,kout,l) = iout
                  i1 = i1 + 1
 100           CONTINUE
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         l1 = l1 + arg_incr(T_AXIS,ARG1)
 400  CONTINUE
      
         
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

