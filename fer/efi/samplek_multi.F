*
* samplek_multi.F
*
* Ansley Manke
* Dec 1998
*
* This function samples data along the K axis using a set of indices
* 4/5/99  Ansley Manke 
*         Indices may be oriented along any axis; 
*         Result is abstract on the sampled axis.
* 4/19  Reverse the order of arguments to be consistent with the
*       SAMPLEL etc functions.
* 4/19  See issue 1924: reworking sample-multi functions

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      SUBROUTINE samplek_multi_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled according to K indices which ',
     .        'may vary in IJLMN')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)

      CALL ef_set_axis_inheritance_6d(id,
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'data to sample using K indices')
      CALL ef_set_axis_influence_6d(id, arg, YES, YES, NO, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'K_INDICES')
      CALL ef_set_arg_desc(id, arg, 'ordered indices')
      CALL ef_set_axis_influence_6d(id, arg, YES, YES, YES, YES, YES, YES)


      RETURN 
      END

* Compute the result
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      SUBROUTINE samplek_multi_compute(id, arg_1, arg_2, result)

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
      INTEGER idim, p1, iexpnd, ishare
      REAL aindex
      CHARACTER adir(6)*1, dir_string*12, errstr*100, dat_shape*8
      DATA adir /'X', 'Y', 'Z', 'T', 'E', 'F'/

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


* If ARG2 doesn't share its axes with ARG1 that's an error

      iexpnd = 0

      DO idim = X_AXIS, F_AXIS

         IF (idim .EQ. Z_AXIS) CYCLE

	 IF (arg_hi_ss(idim, ARG2) .NE. ef_unspecified_int4) THEN
            dir_string = adir(idim)//' direction.'
            IF (arg_hi_ss(idim, ARG2) - arg_lo_ss(idim, ARG2) + 1  .NE. 
     .          arg_hi_ss(idim, ARG1) - arg_lo_ss(idim, ARG1) + 1) GOTO 199

* Also find the first axis that ARG1 and ARG2 share. If arg-1 does not have 
* a Z axis then we're going to expand the result into that direction.
            IF (iexpnd .EQ. 0) iexpnd = idim

	 ENDIF 
	 
      ENDDO

* only need iexpnd if the data-var does not have an E direction
      IF (arg_lo_ss(Z_AXIS,ARG1) .NE. ef_unspecified_int4) iexpnd = 0 

      ishare = 0
      DO idim = X_AXIS, F_AXIS
         IF ( arg_hi_ss(idim, ARG1).NE.ef_unspecified_int4 .AND.
     .        arg_hi_ss(idim, ARG2).NE.ef_unspecified_int4 ) 
     .            ishare = ishare+1

      ENDDO
      IF (ishare .EQ. 0)  GOTO 299  ! no shared axes

* Initialize the result
      result = bad_flag_result

* Now do the sampling

      n1 = arg_lo_ss(F_AXIS, ARG1)
      n2 = arg_lo_ss(F_AXIS, ARG2)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS) 
     
      m1 = arg_lo_ss(E_AXIS, ARG1)
      m2 = arg_lo_ss(E_AXIS, ARG2)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS) 

      l1 = arg_lo_ss(T_AXIS, ARG1)
      l2 = arg_lo_ss(T_AXIS, ARG2)
      DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      DO 300 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      DO 200 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

      k2 = arg_lo_ss(Z_AXIS,ARG2)
      DO 100 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*  Get the index to sample.
*  IF the indices don't lie in range the result is missing.

         aindex = arg_2(i2,j2,k2,l2,m2,n2)
         p1 = aindex

*  Check that we do have an index of ARG_1. Set the result

         IF (aindex .EQ. bad_flag(ARG1)) THEN
            result(i,j,k,l,m,n) = bad_flag_result

         ELSE IF (arg_lo_ss(Z_AXIS,ARG1) .EQ. ef_unspecified_int4) THEN
            k1 = ef_unspecified_int4

            IF (iexpnd .EQ. X_AXIS .AND. 
     .          (p1.GE.arg_lo_ss(X_AXIS,ARG1).AND.p1.LE.arg_hi_ss(X_AXIS,ARG1)) )  
     .          result(i,j,k,l,m,n) = arg_1(p1,j1,k1,l1,m1,n1)

	    IF (iexpnd .EQ. Y_AXIS .AND. 
     .          (p1.GE.arg_lo_ss(Y_AXIS,ARG1).AND.p1.LE.arg_hi_ss(Y_AXIS,ARG1)) ) 
     .          result(i,j,k,l,m,n) = arg_1(i1,p1,k1,l1,m1,n1)

            IF (iexpnd .EQ. Z_AXIS .AND. 
     .          (p1.GE.arg_lo_ss(Z_AXIS,ARG1).AND.p1.LE.arg_hi_ss(Z_AXIS,ARG1)) ) 
     .          result(i,j,k,l,m,n) = arg_1(i1,j1,p1,l1,m1,n1)

            IF (iexpnd .EQ. T_AXIS .AND. 
     .          (p1.GE.arg_lo_ss(T_AXIS,ARG1).AND.p1.LE.arg_hi_ss(T_AXIS,ARG1)) ) 
     .          result(i,j,k,l,m,n) = arg_1(i1,j1,k1,p1,m1,n1)

            IF (iexpnd .EQ. E_AXIS .AND. 
     .          (p1.GE.arg_lo_ss(E_AXIS,ARG1).AND.p1.LE.arg_hi_ss(E_AXIS,ARG1)) ) 
     .          result(i,j,k,l,m,n) = arg_1(i1,j1,k1,p1,m1,n1)

            IF (iexpnd .EQ. F_AXIS .AND. 
     .          (p1.GE.arg_lo_ss(F_AXIS,ARG1).AND.p1.LE.arg_hi_ss(F_AXIS,ARG1)) ) 
     .          result(i,j,k,l,m,n) = arg_1(i1,j1,k1,l1,m1,p1)

         ELSE

            IF (p1 .LT. arg_lo_ss(Z_AXIS,ARG1)  .OR.
     .          p1 .GT. arg_hi_ss(Z_AXIS,ARG1) ) THEN
               result(i,j,k,l,m,n) = bad_flag_result
            ELSE

               IF (arg_1(i1,j1,p1,l1,m1,n1) .EQ. bad_flag(ARG1)) THEN
                  result(i,j,k,l,m,n) = bad_flag_result
               ELSE
                  result(i,j,k,l,m,n) = arg_1(i1,j1,p1,l1,m1,n1)
 
               ENDIF
            ENDIF
         ENDIF

         k2 = k2 + arg_incr(Z_AXIS,ARG2)
 100  CONTINUE

      i1 = i1 + arg_incr(X_AXIS,ARG1)
      i2 = i2 + arg_incr(X_AXIS,ARG2)
 200  CONTINUE

      j1 = j1 + arg_incr(Y_AXIS,ARG1)
      j2 = j2 + arg_incr(Y_AXIS,ARG2)
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

 199  CONTINUE
      errstr = 'Axes of INDEX data must match those in the grid of '//
     .  'DATA_TO_SAMPLE. Mismatch in '//dir_string 
      CALL ef_bail_out(id, errstr)


 299  CONTINUE
      errstr = 'ARG1 has no Z-axis. Add Z dimension to ARG1, e.g.  '//
     .  'DAT_TO_SAMPLE + 0*var_on_z'
      CALL ef_bail_out(id, errstr)

      RETURN
      END
