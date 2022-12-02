*
* dsg_fmask.F
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* Ansley Manke
* Aug 2018
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*
* This function defines a mask on the long, observations axis for variables
* in a Discrete Sampling Geometries ragged-array dataset.  The mask is based
* on a selection of some values of a feature-length variable.


      SUBROUTINE dsg_fmask_str_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_set_desc(id, 
     . 'Returns a mask of length nobs based on values '//
     . 'in feature variable of ragged array dataset')

      CALL ef_set_num_args(id, 4)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id, 
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
c     .                                IMPLIED_BY_ARGS, NORMAL,
c     .                                NORMAL,          NORMAL,
c     .                                NORMAL,          NORMAL)

      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'obsvar')
      CALL ef_set_arg_desc(id, arg, 'A variable on the long, observation axis (1-D)')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)
c     .                              YES, NO, NO, NO, NO, NO)


      arg = 2
      CALL ef_set_arg_name(id, arg, 'rowsize')
      CALL ef_set_arg_desc(id, arg, 'The Rowsize variable, length of each feature')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'Fvar')
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      CALL ef_set_arg_desc(id, arg, 
     . 'A string variable of length feature-number, from which the mask id computed')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)
     

      arg = 4
      CALL ef_set_arg_name(id, arg, 'subset')
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      CALL ef_set_arg_desc(id, arg, 
     . 'Set of strings: mask is 1 where Fvar matches these, else missing.'//
     . 'Mask is true for reatures matching these')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

* * *

      SUBROUTINE dsg_fmask_str_compute(id, var, rowsize, fvar, subset, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL var(*)
      REAL rowsize(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)
      REAL fvar(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .           mem3lot:mem3hit, mem3loe:mem3hie, mem3lof:mem3hif)
      REAL subset(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .           mem3lot:mem3hit, mem3loe:mem3hie, mem3lof:mem3hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)
	
*

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

      LOGICAL ismatch
      INTEGER nobs, sum, nf, nr, ilo, ihi, slen
      INTEGER i,  j,  k,  l,  m,  n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER i4, j4, k4, l4, m4, n4
      CHARACTER*48 fstring, test


      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* Check arguments: For valid DSG files this will have already
* been checked but the function could be called for other data.
* nobs = sum of rowsize. 
* rowsize and fvar must be the same length 
  
      nobs = 0
      DO i =  X_AXIS, F_AXIS
         nobs = MAX(nobs, arg_hi_ss(i,ARG1)-arg_lo_ss(i,ARG1)+1)
      ENDDO

      sum = 0
      nr = 0

      DO i2 = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)
      DO j2 = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)
      DO k2 = arg_lo_ss(Z_AXIS,ARG2), arg_hi_ss(Z_AXIS,ARG2)
      DO l2 = arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2)
      DO m2 = arg_lo_ss(E_AXIS,ARG2), arg_hi_ss(E_AXIS,ARG2)
      DO n2 = arg_lo_ss(F_AXIS,ARG2), arg_hi_ss(F_AXIS,ARG2)
	 IF ( rowsize(i2,j2,k2,l2,m2,n2) .NE. bad_flag(ARG2)) THEN
	    sum = sum + rowsize(i2,j2,k2,l2,m2,n2)
	    nr = nr + 1
	 ELSE
	    GOTO 9100
	 ENDIF
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      IF (sum .NE. nobs) GOTO 9200

      nf = 0
      DO i3 = arg_lo_ss(X_AXIS,ARG3), arg_hi_ss(X_AXIS,ARG3)
      DO j3 = arg_lo_ss(Y_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3)
      DO k3 = arg_lo_ss(Z_AXIS,ARG3), arg_hi_ss(Z_AXIS,ARG3)
      DO l3 = arg_lo_ss(T_AXIS,ARG3), arg_hi_ss(T_AXIS,ARG3)
      DO m3 = arg_lo_ss(E_AXIS,ARG3), arg_hi_ss(E_AXIS,ARG3)
      DO n3 = arg_lo_ss(F_AXIS,ARG3), arg_hi_ss(F_AXIS,ARG3)
         nf = nf + 1
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      IF (nf .NE. nr) GOTO 9300

* Initialize the result
      DO n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
      DO m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
      DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
      DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
      DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         result(i,j,k,l,m,n) = bad_flag_result
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      
* Compute the mask: 1 for all values in features that match the test

      DO i4 = arg_lo_ss(X_AXIS,ARG4), arg_hi_ss(X_AXIS,ARG4)
      DO j4 = arg_lo_ss(Y_AXIS,ARG4), arg_hi_ss(Y_AXIS,ARG4)
      DO k4 = arg_lo_ss(Z_AXIS,ARG4), arg_hi_ss(Z_AXIS,ARG4)
      DO l4 = arg_lo_ss(T_AXIS,ARG4), arg_hi_ss(T_AXIS,ARG4)
      DO m4 = arg_lo_ss(E_AXIS,ARG4), arg_hi_ss(E_AXIS,ARG4)
      DO n4 = arg_lo_ss(F_AXIS,ARG4), arg_hi_ss(F_AXIS,ARG4)
         
	 CALL EF_GET_STRING_ARG_ELEMENT_6D (id, ARG4, subset,
     .                    i4, j4, k4, l4, m4, n4, slen, test)

	 nr = 0
         DO i3 = arg_lo_ss(X_AXIS,ARG3), arg_hi_ss(X_AXIS,ARG3)
         DO j3 = arg_lo_ss(Y_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3)
         DO k3 = arg_lo_ss(Z_AXIS,ARG3), arg_hi_ss(Z_AXIS,ARG3)
         DO l3 = arg_lo_ss(T_AXIS,ARG3), arg_hi_ss(T_AXIS,ARG3)
         DO m3 = arg_lo_ss(E_AXIS,ARG3), arg_hi_ss(E_AXIS,ARG3)
         DO n3 = arg_lo_ss(F_AXIS,ARG3), arg_hi_ss(F_AXIS,ARG3)
             nr = nr + 1
             CALL EF_GET_STRING_ARG_ELEMENT_6D(id, ARG3, fvar,
     .                    i3, j3, k3, l3, m3, n3, slen, fstring)
             IF (test .EQ. fstring) THEN
	        CALL DSG_ROW_LIMITS_SUB (rowsize, nr, ilo, ihi)
		CALL PUT_ROW_MASK_SUB (result, ilo, ihi)
		GOTO 444
             ENDIF
         ENDDO
         ENDDO
         ENDDO
         ENDDO
         ENDDO
         ENDDO

  444	 CONTINUE
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      RETURN

 9100 CALL EF_BAIL_OUT (id, 'The Rowsize variable may not contain missing data')
 9200 CALL EF_BAIL_OUT (id, 'The sum of Rowsize must equal the length of argument 1')
 9300 CALL EF_BAIL_OUT (id, 'The length of the Rowsize and Fvar must be equal')

      END

      SUBROUTINE DSG_ROW_LIMITS_SUB (rowsize, nr, ilo, ihi)

* Return the low and high index limits for row nr of rowsize 

      REAL    rowsize(*)
      INTEGER nr, ilo, ihi

      INTEGER i

      IF (nr .EQ. 1) THEN
         ilo = 1
	 ihi = rowsize(1)
      ELSE

         ilo = 1
         DO i = 2, nr
            ilo = ilo + rowsize(i-1) 
	 ENDDO
	 ihi = ilo + rowsize(nr)
      ENDIF

      RETURN
      END

      SUBROUTINE PUT_ROW_MASK_SUB (result, ilo, ihi)

* Put 1 into the result for the index limits, avoids all the
* 6-D looping.

      REAL result(*)
      INTEGER ilo, ihi

      INTEGER i
      DO i = ilo, ihi
         result(i) = 1.
      ENDDO

      RETURN
      END
