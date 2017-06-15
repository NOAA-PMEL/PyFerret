*
* samplen_multi.F
*
* Ansley Manke
* Dec 1998
*
* This function samples data along the L axis using a set of indices
* 4/5/99  Ansley Manke 
*         Indices may be oriented along any axis; 
*         Result is abstract on the sampled axis.
*
* 5/17  Make these functions 6-D.  Add E and F direction functions.
*       Reverse the order of argumnets to beconsistent with the
*       SAMPLEL etc functions. Fix the test for sample argument on 
*       different axis than the sample direction.
*

      SUBROUTINE samplen_multi_init(id)

      INCLUDE 'EF_Util.cmn'

* Declarations
      INTEGER id, arg
      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled according to N indices which ',
     . 'may be function of IJKLM')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, ABSTRACT, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'data to sample using L indices')
      CALL ef_set_axis_influence_6d(id, arg, YES, YES, YES, YES, YES, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'N_INDICES')
      CALL ef_set_arg_desc(id, arg, 'ordered indices')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      RETURN 
      END

* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.  

      SUBROUTINE samplen_multi_result_limits(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id
      INTEGER arg_lo_ss(6,EF_MAX_ARGS), arg_hi_ss(6,EF_MAX_ARGS),
     .     arg_incr(6,EF_MAX_ARGS)

      INTEGER npts, idir, ndim
      CHARACTER*100 errtxt

*     Use utility functions to get context information about the 
*     1st argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      npts = arg_hi_ss(F_AXIS, ARG2) - arg_lo_ss(F_AXIS, ARG2) + 1

* Count the dimensions in argument 2
      ndim = 0
      DO idir = X_AXIS, F_AXIS
         IF (arg_lo_ss(idir, ARG2) .NE. ef_unspecified_int4) THEN
	    ndim = ndim + 1
         ENDIF
      ENDDO

* If they sent in a simple list as the 2nd argument, it might not be 
* on the F axis. Allow that.

      IF (arg_lo_ss(F_AXIS, ARG2) .EQ. ef_unspecified_int4) THEN
         ndim = 0
         DO idir = X_AXIS, F_AXIS
	    IF (arg_lo_ss(idir, ARG2) .NE. ef_unspecified_int4) THEN
	       npts = arg_hi_ss(idir, ARG2) - arg_lo_ss(idir, ARG2) + 1
	       ndim = ndim + 1
	    ENDIF
         ENDDO
	 IF (ndim .NE. 1) THEN
	    errtxt = 'Argument 2 has no F axis. When it has more than '//
     .               '1 dimension, F axis is the sample dimension.' 
	    GOTO 999
	 ENDIF

* If argument 2 is a multi-dimensioned variable, all but the sample dimension
* must match the grid of argument 1
      ELSE
         IF (ndim .GT. 1) THEN
         DO idir = X_AXIS, E_AXIS
	   IF (arg_hi_ss(idir,ARG1)-arg_lo_ss(idir,ARG1) .NE. 
     .         arg_hi_ss(idir,ARG2)-arg_lo_ss(idir,ARG2) ) THEN
	      errtxt = 'Grid mismatch. Non-F dimensions of Arg 2 must be '//
     .               'conformable with grid of Arg 1' 
	      GOTO 999
	   ENDIF
         ENDDO
	 ENDIF
      ENDIF

      CALL ef_set_axis_limits(id, F_AXIS, 1, npts)
      RETURN
  999 CONTINUE

      CALL ef_bail_out(id, 'Argument 2 hsd no F axis. '//
     .  'If it has >1 dimension, F axis is the sample dimension.' )
 
      END
*
* In this subroutine we compute the result
*
      SUBROUTINE samplen_multi_compute(id, arg_1, arg_2, result)

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

* Local variable declarations
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER norder
      REAL aorder

      INTEGER nlen(3)
      CHARACTER*8 lefint, aindex(3)
      CHARACTER*100 errtxt

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


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

*  Get the index to sample, whichever axis of arg_2 it is on.

                  aorder = arg_2(i2,j2,k2,l2,m2,n2)
                  IF (n2 .EQ. ef_unspecified_int4) 
     .                  CALL pickindex(arg_2, n, aorder)
                  norder = aorder

*  Check that we do have an index of ARG_1; set the result

                  IF (aorder .EQ. bad_flag(ARG2)) THEN
                     result(i,j,k,l,m,n) = bad_flag_result
                  ELSE

                     IF (norder .LT. arg_lo_ss(F_AXIS,ARG1)  .OR.
     .                   norder .GT. arg_hi_ss(F_AXIS,ARG1) ) THEN
                        GO TO 999
                     ELSE

                        IF (arg_1(i1,j1,k1,m1,n1,norder) .EQ. bad_flag(ARG1)) THEN
                           result(i,j,k,l,m,n) = bad_flag_result
                        ELSE
                           result(i,j,k,l,m,n) = arg_1(i1,j1,k1,m1,n1,norder)
 
                        END IF
                     END IF
                  END IF

                  n2 = n2 + arg_incr(F_AXIS,ARG2)
 100           CONTINUE

               i1 = i1 + arg_incr(X_AXIS,ARG1)
               i2 = i2 + arg_incr(X_AXIS,ARG2)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300     CONTINUE

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

999   CONTINUE

      aindex(1) = LEFINT(norder,nlen(1))
      aindex(2) = LEFINT(arg_lo_ss(T_AXIS,ARG2),nlen(2))
      aindex(3) = LEFINT(arg_hi_ss(T_AXIS,ARG2),nlen(3))

      WRITE (errtxt,*) 'Sampling index ', aindex(1)(1:nlen(1)), 
     . ' in ARG1 is outside the range of L indices for ARG2 (', 
     .  aindex(2)(1:nlen(2)), ':', aindex(3)(1:nlen(3)), ')'

      CALL EF_BAIL_OUT(id, errtxt)

      RETURN
      END

