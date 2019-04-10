*
* samplen_multi.F
*

* Ansley Manke
*
* 5/17  Make these functions 6-D.  Add E and F direction functions.
*       Reverse the order of arguments to be consistent with the
*       SAMPLEL etc functions. Fix the test for sample argument on 
*       different axis than the sample direction.
* 4/19  See issue 1924: reworking sample-multi functions
*
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      SUBROUTINE samplen_multi_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

* Declarations
      INTEGER id, arg
      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled according to N indices which ',
     .        'may vary in IJKLM')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'data to sample using N indices')
      CALL ef_set_axis_influence_6d(id, arg, YES, YES, YES, YES, YES, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'N_INDICES')
      CALL ef_set_arg_desc(id, arg, 'ordered indices')
      CALL ef_set_axis_influence_6d(id, arg, YES, YES, YES, YES, YES, YES)

      RETURN 
      END

* Compute the result
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      SUBROUTINE samplen_multi_compute(id, arg_1, arg_2, result)

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
      INTEGER idim, norder, iexpnd
      REAL aorder
      CHARACTER adir(6)*1, dir_string*12, errstr*100
      DATA adir /'X', 'Y', 'Z', 'T', 'E', 'F'/

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* If ARG2 doesn't share its axes with ARG1 that's an error

      iexpnd = 0

      DO idim = X_AXIS, F_AXIS

         IF (idim .EQ. F_AXIS) CYCLE

	 IF (arg_hi_ss(idim, ARG2) .NE. ef_unspecified_int4) THEN
            dir_string = adir(idim)//' direction.'
            IF (arg_hi_ss(idim, ARG2) - arg_lo_ss(idim, ARG2) + 1  .NE. 
     .          arg_hi_ss(idim, ARG1) - arg_lo_ss(idim, ARG1) + 1) GOTO 199

* Also find the first axis that ARG1 and ARG2 share. If arg-1 does not have 
* an F axis then we're going to expand the result into that direction.
            IF (iexpnd .EQ. 0) iexpnd = idim

	 ENDIF 
	 
      ENDDO

* only need iexpnd if the data-var does not have an E direction
      IF (arg_lo_ss(F_AXIS,ARG1) .NE. ef_unspecified_int4) iexpnd = 0 

      m1 = arg_lo_ss(E_AXIS, ARG1)
      m2 = arg_lo_ss(E_AXIS, ARG2)
      DO 600 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS) 

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

      n2 = arg_lo_ss(F_AXIS,ARG2)
      DO 100 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

*  Get the index to sample.
*  IF the indices don't lie in range the result is missing.

         aorder = arg_2(i2,j2,k2,l2,m2,n2)
         norder = aorder

*  Check that we do have an index of ARG_1. Set the result

         IF (aorder .EQ. bad_flag(ARG2)) THEN
            result(i,j,k,l,m,n) = bad_flag_result

         ELSE IF (arg_lo_ss(F_AXIS,ARG1) .EQ. ef_unspecified_int4) THEN
            n1 = ef_unspecified_int4
            IF (iexpnd .EQ. X_AXIS) result(i,j,k,l,m,n) = arg_1(norder,j1,k1,l1,m1,n1)
            IF (iexpnd .EQ. Y_AXIS) result(i,j,k,l,m,n) = arg_1(i1,norder,k1,l1,m1,n1)
            IF (iexpnd .EQ. Z_AXIS) result(i,j,k,l,m,n) = arg_1(i1,j1,norder,l1,m1,n1)
            IF (iexpnd .EQ. T_AXIS) result(i,j,k,l,m,n) = arg_1(i1,j1,k1,norder,m1,n1)
            IF (iexpnd .EQ. E_AXIS) result(i,j,k,l,m,n) = arg_1(i1,j1,k1,l1,norder,n1)
         ELSE

            IF (norder .LT. arg_lo_ss(F_AXIS,ARG1)  .OR.
     .          norder .GT. arg_hi_ss(F_AXIS,ARG1) ) THEN
               result(i,j,k,l,m,n) = bad_flag_result
            ELSE

               IF (arg_1(i1,j1,k1,l1,m1,norder) .EQ. bad_flag(ARG1)) THEN
                  result(i,j,k,l,m,n) = bad_flag_result
               ELSE
                  result(i,j,k,l,m,n) = arg_1(i1,j1,k1,l1,m1,norder)
               ENDIF
            ENDIF
         ENDIF

         n2 = n2 + arg_incr(F_AXIS,ARG2)
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

      m1 = m1 + arg_incr(E_AXIS,ARG1)
      m2 = m2 + arg_incr(E_AXIS,ARG2)
 600  CONTINUE

      RETURN

 199  CONTINUE
      errstr = 'Axes of INDEX data must match those in the grid of '//
     .  'DATA_TO_SAMPLE. Mismatch in '//dir_string 
      CALL ef_bail_out(id, errstr)

      RETURN
      END
