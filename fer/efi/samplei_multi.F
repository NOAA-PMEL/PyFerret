*
* samplei_multi.F
*
* Ansley Manke
* Dec 1998
*
* This function samples data along the I axis using a set of indices
* which may be different for the j,k,L,m,n dimensions.
* 4/5/99  Ansley Manke 
*         Indices may be oriented along any axis; 
*         Result is abstract on the sampled axis.
*
* 4/19  Make these functions 6-D. 
*       Reverse the order of arguments to be consistent with the
*       SAMPLEL etc functions.
*
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      SUBROUTINE samplei_multi_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled according to I indices which ',
     .        'may vary in JKLMN')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2) 

      CALL ef_set_axis_inheritance_6d(id,
     .            ABSTRACT, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'data to sample using I indices')
      CALL ef_set_axis_influence_6d(id, arg, NO, YES, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'I_INDICES')
      CALL ef_set_arg_desc(id, arg, 'ordered indices')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      RETURN 
      END

      

* Set the lo and hi limits associated with each abstract result axis. 

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      SUBROUTINE samplei_multi_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id
      INTEGER arg_lo_ss(6,EF_MAX_ARGS), arg_hi_ss(6,EF_MAX_ARGS),
     .     arg_incr(6,EF_MAX_ARGS)
      INTEGER lo, hi
      INTEGER nn, count, idim

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

* The X axis of the indices is the index length if it exists, 
* or find the axis of a 1D list

      lo = 1
      hi = arg_hi_ss(X_AXIS, ARG2) - arg_lo_ss(X_AXIS, ARG2) + 1

* If the index list is empty on the X axis, then see if it's a simple 1D list
* in another direction.

      IF (hi .EQ. 1)  THEN
         count = 0
	 DO idim = Y_AXIS, F_AXIS

            nn = arg_hi_ss(idim, ARG2) - arg_lo_ss(idim, ARG2) + 1
	    IF (nn .GT. 1) THEN
	       hi = nn
	       count = count + 1
	       IF (count .GT. 1) GOTO 999
	    ENDIF
	 ENDDO
      ENDIF


      CALL ef_set_axis_limits(id, X_AXIS, lo, hi)

      RETURN
      
999   CONTINUE
      CALL ef_bail_out(id, 
     .  'Argument 2: Index list must have an X axis or be one-dimensional')

      RETURN 
      END
*


* Compute the result

** * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      SUBROUTINE samplei_multi_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

* Declarations
      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .      mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit,
     .      memresloe:memreshie, memreslof:memreshif)


      INTEGER res_lo_ss(6), res_hi_ss(6), res_incr(6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS), arg_hi_ss(6,EF_MAX_ARGS),
     .     arg_incr(6,EF_MAX_ARGS)

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER iorder
      REAL aorder

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


      n1 = arg_lo_ss(F_AXIS, ARG1)
      n2 = arg_lo_ss(F_AXIS, ARG2)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS) 
     
      m1 = arg_lo_ss(E_AXIS, ARG1)
      m2 = arg_lo_ss(E_AXIS, ARG2)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS) 

      l1 = arg_lo_ss(T_AXIS, ARG1)
      l2 = arg_lo_ss(T_AXIS, ARG2)
      DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 

      k1 = arg_lo_ss(Z_AXIS, ARG1)
      k2 = arg_lo_ss(Z_AXIS, ARG2)
      DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS) 

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

      i2 = arg_lo_ss(X_AXIS,ARG2)
      DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

*  Get the index to sample, whichever axis of arg_1 it is on.

         aorder = arg_2(i2,j2,k2,l2,m2,n2)
         IF (i2 .EQ. ef_unspecified_int4) CALL pickindex(arg_2, i, aorder)
         iorder = aorder

*  Check that we do have an index of ARG_2. Set the result

         IF (aorder .EQ. bad_flag(ARG1)) THEN
            result(i,j,k,l,m,n) = bad_flag_result
         ELSE

            IF (iorder .LT. arg_lo_ss(X_AXIS,ARG1)  .OR.
     .          iorder .GT. arg_hi_ss(X_AXIS,ARG1) ) THEN
               result(i,j,k,l,m,n) = bad_flag_result
            ELSE

               IF (arg_1(iorder,j1,k1,l1,m1,n1) .EQ. bad_flag(ARG1)) THEN
                  result(i,j,k,l,m,n) = bad_flag_result
               ELSE
                  result(i,j,k,l,m,n) = arg_1(iorder,j1,k1,l1,m1,n1)
               ENDIF
            
	    ENDIF

         ENDIF

         i2 = i2 + arg_incr(X_AXIS,ARG2)
 100  CONTINUE

      j1 = j1 + arg_incr(Y_AXIS,ARG1)
      j2 = j2 + arg_incr(Y_AXIS,ARG2)
 200  CONTINUE

      k1 = k1 + arg_incr(Z_AXIS,ARG1)
      k2 = k2 + arg_incr(Z_AXIS,ARG2)
 300  CONTINUE

      l1 = l1 + arg_incr(T_AXIS,ARG1)
      l2 = l2 + arg_incr(T_AXIS,ARG2)
 400  CONTINUE

      m1 = m1 + arg_incr(E_AXIS,ARG1)
      m2 = m2 + arg_incr(E_AXIS,ARG2)
 500  CONTINUE

      n1 = n1 + arg_incr(F_AXIS,ARG1)
      n2 = n2 + arg_incr(F_AXIS,ARG2)
 600  CONTINUE


      RETURN
      END

