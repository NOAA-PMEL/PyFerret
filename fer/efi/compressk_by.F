* 
*  compressk_by.F
* 
*  Steve Hankin
*  12/27/2000
* 
*  Compress argument 1 along axis K by the mask given in argument 2
*  Argument 2 may be multidimensional
 
*  In this subroutine we provide information about
*  the function.  The user configurable information 
*  consists of the following:
* 
*  descr              Text description of the function
* 
*  num_args           Required number of arguments
* 
*  axis_inheritance   Type of axis for the result
*                        ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                        CUSTOM          - user defined axis
*                        IMPLIED_BY_ARGS - same axis as the incoming argument
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


      SUBROUTINE compressk_by_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,'Compress arg1 by (multi-D) mask in arg 2' )

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, ABSTRACT, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'dat')
      CALL ef_set_arg_desc(id, arg, 'variable to compress on K axis')
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'mask')
      CALL ef_set_arg_desc(id, arg,
     .          'valid/bad points guiding compression (multi-D)')
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, YES)
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

      SUBROUTINE compressk_by_result_limits(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_hi_k, lmsk
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the arguments.
*

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      my_hi_k = arg_hi_ss(Z_AXIS,1) - arg_lo_ss(Z_AXIS,1) + 1

      lmsk    = arg_hi_ss(Z_AXIS,2) - arg_lo_ss(Z_AXIS,2) + 1
      IF ( lmsk .LT. my_hi_k ) my_hi_k = lmsk

      CALL ef_set_axis_limits(id, Z_AXIS, 1, my_hi_k)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



* 
*  In this subroutine we compute the result
* 
      SUBROUTINE compressk_by_compute(id, arg_1, arg_2, result)

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
      INTEGER i2, j2, k2, l2
      INTEGER nout, klo

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      klo = res_lo_ss(Z_AXIS)

* pre-fill the entire output result with missing value flags
*  (since each K-sequence in mask may have different number of valid pts)
      DO 40 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         DO 30 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            DO 20 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
               DO 10 k=klo, res_hi_ss(Z_AXIS)
 10            result(i,j,k,l) = bad_flag_result
 20         CONTINUE
 30      CONTINUE
 40   CONTINUE

* fill in the points as specified by the mask
      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            l2 = arg_lo_ss(T_AXIS,ARG2)
            DO 200 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

               nout = 0
               k1 = arg_lo_ss(Z_AXIS,ARG1)
               k2 = arg_lo_ss(Z_AXIS,ARG2)
               DO 100 k=klo, res_hi_ss(Z_AXIS)

                  IF ( arg_2(i2,j2,k2,l2) .NE. bad_flag(2) ) THEN
                     IF ( arg_1(i1,j1,k1,l1) .EQ. bad_flag(1)  ) THEN
                        result(i,j,klo+nout,l) = bad_flag_result
                     ELSE
                        result(i,j,klo+nout,l) = arg_1(i1,j1,k1,l1) 
                     END IF
                     nout = nout + 1
                  ENDIF

                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
                  k2 = k2 + arg_incr(Z_AXIS,ARG2)
 100           CONTINUE

               l1 = l1 + arg_incr(T_AXIS,ARG1)
               l2 = l2 + arg_incr(T_AXIS,ARG2)
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

