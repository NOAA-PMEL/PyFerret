*
* samplel.F
*
* Ansley Manke
* Dec 1998
*
* This function samples data along the L axis using a set of indices
* 4/5/99  Ansley Manke 
*         Indices may be oriented along any axis; 
*         Result is abstract on the sampled axis.
*    6-Jul-1999  change order of the arguments: data-to-sample, indices
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


      SUBROUTINE samplel_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled according to a set of L indices')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, ABSTRACT)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'data to sample using L indices')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'L_INDICES')
      CALL ef_set_arg_desc(id, arg, 'ordered indices')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
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
* loss               lo subscript for an axis
*
* hiss               hi subscript for an axis
*

      SUBROUTINE samplel_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo_l, my_hi_l
      INTEGER nx, ny, nz, nt

*     Use utility functions to get context information about the 
*     2nd argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG2) - arg_lo_ss(X_AXIS, ARG2) + 1
      ny = arg_hi_ss(Y_AXIS, ARG2) - arg_lo_ss(Y_AXIS, ARG2) + 1
      nz = arg_hi_ss(Z_AXIS, ARG2) - arg_lo_ss(Z_AXIS, ARG2) + 1
      nt = arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) + 1

*  If there is a list of indices along the T-axis of ARG1, use the
*  argument subscripts for the abstract axis limits.

      my_lo_l = 1
      my_hi_l = nt

      IF (arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) .EQ. 0) 
     .    my_hi_l = MAX(nx,ny,nz,nt)

      CALL ef_set_axis_limits(id, T_AXIS, my_lo_l, my_hi_l)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
*
* In this subroutine we compute the result
*
      SUBROUTINE samplel_compute(id, arg_1, arg_2, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .      mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

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
      INTEGER i, j, k, l
      INTEGER i1, j1, k1
      INTEGER i2, j2, k2, l2
      INTEGER lorder
      REAL aorder

      INTEGER nlen(3), unspecified_int4
      CHARACTER*8 lefint, aindex(3)
      CHARACTER*100 errtxt

      unspecified_int4 = -111

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


      i1 = arg_lo_ss(X_AXIS, ARG1)
      i2 = arg_lo_ss(X_AXIS, ARG2)
      DO 400 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS) 

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         DO 300 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            k2 = arg_lo_ss(Z_AXIS,ARG2)
            DO 200 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l2 = arg_lo_ss(T_AXIS,ARG2)
               DO 100 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

*  Get the index to sample, whichever axis of arg_1 it is on.

                  aorder = arg_2(i2,j2,k2,l2)
                  IF (l2 .EQ. unspecified_int4) 
     .                  CALL pickindex(arg_2, l, aorder)
                  lorder = aorder

*  Check that we do have an index of ARG_2; set the result

                  IF (aorder .EQ. bad_flag(ARG2)) THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSE

                     IF (lorder .LT. arg_lo_ss(T_AXIS,ARG1)  .OR.
     .                   lorder .GT. arg_hi_ss(T_AXIS,ARG1) ) THEN
                        GO TO 999
                     ELSE

                        IF (arg_1(i1,j1,k1,lorder) .EQ. bad_flag(ARG1))
     .                   THEN
                           result(i,j,k,l) = bad_flag_result
                        ELSE
                           result(i,j,k,l) = arg_1(i1,j1,k1,lorder)
 
                        END IF
                     END IF
                  END IF

                  l2 = l2 + arg_incr(T_AXIS,ARG2)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
               k2 = k2 + arg_incr(Z_AXIS,ARG2)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
 400  CONTINUE

      RETURN 

999   CONTINUE

      aindex(1) = LEFINT(lorder,nlen(1))
      aindex(2) = LEFINT(arg_lo_ss(T_AXIS,ARG1),nlen(2))
      aindex(3) = LEFINT(arg_hi_ss(T_AXIS,ARG1),nlen(3))

      WRITE (errtxt,*) 'Sampling index ', aindex(1)(1:nlen(1)), 
     . ' in ARG2 is outside the range of L indices for ARG1 (', 
     .  aindex(2)(1:nlen(2)), ':', aindex(3)(1:nlen(3)), ')'

      CALL EF_BAIL_OUT(id, errtxt)

      RETURN
      END
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************
      SUBROUTINE pickindex(arr, lindex, aorder)

*  Get the index to sample, whichever axis of arg_1 it is on.  Collapse
*  the array to one dimension and return the l'th point.

      real arr(*), aorder
      integer lindex

      aorder = arr(lindex)

      return
      end
