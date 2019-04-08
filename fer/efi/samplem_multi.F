*
* samplem_multi.F
*
* Ansley Manke
*
* 5/17  Make these functions 6-D.  Add E and F direction functions.
* 4/19  See issue 1924: reworking sample-multi functions
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      SUBROUTINE samplem_multi_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

     
* Declarations
      INTEGER id, arg

      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled according to M indices which ',
     .        'may vary in IJKLN')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS,          CUSTOM, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'data to sample using M indices')
      CALL ef_set_axis_influence_6d(id, arg, YES, YES, YES, YES, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'M_INDICES')
      CALL ef_set_arg_desc(id, arg, 'ordered indices')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      RETURN 
      END

* Define the simple custom axis
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

      SUBROUTINE samplem_multi_custom_axes(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id
      INTEGER arg_lo_ss(6,EF_MAX_ARGS), arg_hi_ss(6,EF_MAX_ARGS),
     .     arg_incr(6,EF_MAX_ARGS)

      REAL lo, hi
      INTEGER idim

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

* The E axis of the indices is the index length

      lo = 1
      hi = arg_hi_ss(E_AXIS, ARG2) - arg_lo_ss(E_AXIS, ARG2) + 1

* If ARG2 doesn't share its axes with ARG1 that's an error

      DO idim = 1, F_AXIS
	 IF (arg_hi_ss(idim, ARG2) .NE. ef_unspecified_int4) THEN
	    IF (arg_hi_ss(idim, ARG1) .EQ. ef_unspecified_int4) GOTO 999
	 ENDIF
      ENDDO

      CALL ef_set_custom_axis(id, E_AXIS, lo, hi, 1.0, 'Index', NO)

      RETURN

999   CONTINUE
      CALL ef_bail_out(id, 
     .  'ARG2: Grid of index list must be a subset of the grid of the data being sampled')
      
      RETURN
      END

*
* Compute the result
** * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

      SUBROUTINE samplem_multi_compute(id, arg_1, arg_2, result)

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
      INTEGER morder
      REAL aorder

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


      n1 = arg_lo_ss(F_AXIS, ARG1)
      n2 = arg_lo_ss(F_AXIS, ARG2)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS) 

      l1 = arg_lo_ss(T_AXIS, ARG1)
      l2 = arg_lo_ss(T_AXIS, ARG2)
      DO 500 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 

      k1 = arg_lo_ss(Z_AXIS, ARG1)
      k2 = arg_lo_ss(Z_AXIS, ARG2)
      DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS) 

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      DO 300 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      DO 200 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

      m2 = arg_lo_ss(E_AXIS,ARG2)
      DO 100 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

*  Get the index to sample.
*  IF the indices don't lie in range the result is missing.
*  (This is normally handled by having the inheritance depend on both ARG1 and ARG2
*   but that checking is done on coordinates and here we are working with indices,
*   and here we allow a return - which may be missing.)

         IF (i2.LT.arg_lo_ss(X_AXIS,ARG2) .OR. i2.GT.arg_hi_ss(X_AXIS,ARG2) .OR. 
     .       j2.LT.arg_lo_ss(Y_AXIS,ARG2) .OR. j2.GT.arg_hi_ss(Y_AXIS,ARG2) .OR. 
     .       k2.LT.arg_lo_ss(Z_AXIS,ARG2) .OR. k2.GT.arg_hi_ss(Z_AXIS,ARG2) .OR. 
     .       l2.LT.arg_lo_ss(T_AXIS,ARG2) .OR. l2.GT.arg_hi_ss(T_AXIS,ARG2) .OR. 
     .       m2.LT.arg_lo_ss(E_AXIS,ARG2) .OR. m2.GT.arg_hi_ss(E_AXIS,ARG2) .OR. 
     .       n2.LT.arg_lo_ss(F_AXIS,ARG2) .OR. n2.GT.arg_hi_ss(F_AXIS,ARG2) ) THEN

            result(i,j,k,l,m,n) = bad_flag_result
            m2 = m2 + arg_incr(E_AXIS,ARG2)
	    CYCLE
         ENDIF

         aorder = arg_2(i2,j2,k2,l2,m2,n2)
         morder = aorder

*  Check that we do have an index of ARG_1. Set the result

         IF (aorder .EQ. bad_flag(ARG2)) THEN
            result(i,j,k,l,m,n) = bad_flag_result
         ELSE

            IF (morder .LT. arg_lo_ss(E_AXIS,ARG1)  .OR.
     .          morder .GT. arg_hi_ss(E_AXIS,ARG1) ) THEN
               result(i,j,k,l,m,n) = bad_flag_result
            ELSE

               IF (arg_1(i1,j1,k1,l1,morder,n1) .EQ. bad_flag(ARG1)) THEN
                  result(i,j,k,l,m,n) = bad_flag_result
               ELSE
                  result(i,j,k,l,m,n) = arg_1(i1,j1,k1,l1,morder,n1)
               ENDIF
            ENDIF
         ENDIF

         m2 = m2 + arg_incr(E_AXIS,ARG2)
 100  CONTINUE

      i1 = i1 + arg_incr(X_AXIS,ARG1)
      i2 = i2 + arg_incr(X_AXIS,ARG2)
 200  CONTINUE

      j1 = j1 + arg_incr(Y_AXIS,ARG1)
      j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300  CONTINUE

      k1 = k1 + arg_incr(Z_AXIS,ARG1)
      k2 = k2 + arg_incr(Z_AXIS,ARG2)
 400  CONTINUE
   
      l1 = l1 + arg_incr(T_AXIS,ARG1)
      l2 = l2 + arg_incr(T_AXIS,ARG2)
 500  CONTINUE

      n1 = n1 + arg_incr(F_AXIS,ARG1)
      n2 = n2 + arg_incr(F_AXIS,ARG2)
 600  CONTINUE

      RETURN
      END

