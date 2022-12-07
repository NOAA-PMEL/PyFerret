* 
*  expndi_by_z_counts.F
* 
*  Ansley Manke (from expndi_by_z)
*  8/24/2011
* 
* Expand argument 1 by creating a 2D XZ structure from it with the 
* separation between Z columns given by the lengths specified in 
* argument 2. Argument 2 is a list of length N, where N is the number of 
* Z columns to create, and each item in the list is the length of that 
* column (the length of the cruise or profile). Thus the Z columns created 
* are "ragged" Typical use is that arg 1 is a collection of vertical casts 
* appended end-to-end and that arg 2 is the number of values in each cast.
* The sum of the values in arg 2 is the length of argument 1. 

*  Args 3 and 4 are constants -- the size of the X and Z axes, respectively
 
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


      SUBROUTINE expndi_by_z_counts_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .  'Expand arg1 on I into an XZ grid. (Typ. to split Z-profiles)' )

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .     IMPLIED_BY_ARGS, ABSTRACT, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'dat')
      CALL ef_set_arg_desc(id, arg, 'variable to expand on I axis')
      CALL ef_set_axis_influence(id, arg, NO, YES, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'counts')
      CALL ef_set_arg_desc(id, arg,
     .          'count of values in each profile')
      CALL ef_set_axis_influence(id, arg, NO, YES, NO, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'max_profile_len')
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

      SUBROUTINE expndi_by_z_counts_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      REAL size
      integer nx, arg
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the arguments.
*
* output X axis size
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      arg = 2
      nx = arg_hi_ss(X_AXIS,arg) - arg_lo_ss(X_AXIS,arg) + 1
      CALL ef_set_axis_limits(id, X_AXIS, 1, nx)

* output Z axis size
      CALL ef_get_one_val(id, ARG3, size)
      CALL ef_set_axis_limits(id, Z_AXIS, 1, INT(ABS(size)))
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



* 
*  In this subroutine we compute the result
* 
      SUBROUTINE expndi_by_z_counts_compute(id, arg_1, arg_2, arg_3, 
     .                               result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy,
     .	   mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .     mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy,
     .     mem3loz:mem3hiz, mem3lot:mem3hit)
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
      INTEGER i2, j2, k2, l2
      INTEGER imax,kmax, iout, kout, isum

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* limits on output arrays
        imax = res_hi_ss(X_AXIS)
        kmax = res_hi_ss(Z_AXIS)

* make sure input 2 is a list of length nprofile
      IF (arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1 .NE. imax)
     .       CALL EF_BAIL_OUT(id, 'Arg 2 must be of length given in arg3')

* args 1 and 2 must not depend on z
      IF ( (arg_lo_ss(Z_AXIS,ARG1) .NE. arg_hi_ss(Z_AXIS,ARG1))
     . .OR.(arg_lo_ss(Z_AXIS,ARG2) .NE. arg_hi_ss(Z_AXIS,ARG2)) ) CALL
     .           EF_BAIL_OUT(id, 'Z size of args 1 and 2 must be 1')

* sum of counts in arg 2 must be length of x-dimension of arg1

      k2 = arg_lo_ss(Z_AXIS,ARG2)
      do j2 = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)
      do L2 = arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2)
         isum = 0
         DO i2 = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)
            IF  (arg_2(i2,j2,k2,l2) .EQ. bad_flag(ARG2) .OR. 
     .           INT(arg_2(i2,j2,k2,l2)) .LT. 0)
     .          CALL EF_BAIL_OUT(id, 'elements of arg 2 are counts GE zero')
            isum = isum + INT(arg_2(i2,j2,k2,l2))
         enddo
         IF (isum .NE. (arg_hi_ss(X_AXIS,ARG1)-arg_lo_ss(X_AXIS,ARG1)+1) )
     .       CALL EF_BAIL_OUT(id, 
     .         'arg 2 values must sum to x-dimension length of of arg1 ')
      enddo
      enddo

* pre-fill the entire output result with missing value flags
*  (since length of each profile may have different number of pts)
      result(i,j,k,l) = bad_flag_result

* fill in the points as specified by the counts
      k1 = arg_lo_ss(Z_AXIS,ARG1)
      k2 = arg_lo_ss(Z_AXIS,ARG2)

      l1 = arg_lo_ss(T_AXIS,ARG1)
      l2 = arg_lo_ss(T_AXIS,ARG2)
      DO 400 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            i1 = 1
            DO 200 iout=arg_lo_ss(X_AXIS,ARG2),arg_hi_ss(X_AXIS,ARG2)
               DO 100 kout = 1, NINT(arg_2(iout,j2,k2,l2))
                 IF ( arg_1(i1,j1,k1,l1) .NE. bad_flag(1) )
     .             result(iout,j,kout,l) = arg_1(i1,j1,k1,l1)
                  i1 = i1 + 1
 100           CONTINUE
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300     CONTINUE

         l1 = l1 + arg_incr(T_AXIS,ARG1)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
 400  CONTINUE
      
         
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

