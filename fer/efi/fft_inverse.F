*------------------------------fft_inverse.F---------------------------

* fft_inverse.F
*
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
* Ansley Manke
* November 2001
*
* v5.50  *acm* 10/02 Simplify the error message for missing data when the 
*                    input is a simple 1-D time series.
C  11/2022 *acm* Code cleanup: Update FFTPACK code to F90 standard, from
C                https://github.com/fortran-lang/fftpack.git 
*                Send an integer work array along with wft tom avoid 
*                type-mismatch warnings
*
* This function computes inverse fft transform spectrum for each A,B,
* where these are the real and imaginary part of the FFT coefficients.
*
*  NOTE:
*  IT IS GENERALLY ADVISABLE TO INCLUDE EXPLICIT LIMITS WHEN WORKING WITH
*  FUNCTIONS THAT REPLACE AXES. FOR EXAMPLE, THE CONSIDER THE FUNCTION
*  SORTL(v). THE EXPRESSION
*       LIST/L=6:10 SORTL(v)
*  IS NOT EQUIVALENT TO
*       LIST SORTL(v[L=6:10])
*  THE FORMER WILL LIST THE 6TH THROUGH 10TH SORTED INDICES FROM THE ENTIRE
*  L RANGE OF VARIABLE V. THE LATTER WILL LIST ALL OF THE INDICES THAT
*  RESULT FROM SORTING v[L=6:10].
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


      SUBROUTINE fft_inverse_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 'Computes inverse fft time series')
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, ABSTRACT,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'Real part of FFT coefficients')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 
     .                 'Imaginary part of FFT coefficients')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, NO, YES, YES)

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
* lo_ss               lo subscript for an axis
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE fft_inverse_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER ntime, nfreq
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

*  Use utility functions to get context information about the argument.

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nfreq = arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1) + 1

C  The FFT code returns frequencies W(k) for k=1 to N/2, with N/2 rounded down

      ntime = nfreq* 2

      CALL ef_set_axis_limits(id, T_AXIS, 1, ntime)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE fft_inverse_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER mtdat, mtwork
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mtdat = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

      mtwork = 4.* 2.5* mtdat + 15

* a
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                            mtdat, 1, 1, 1, 1, 1)

* b
      CALL ef_set_work_array_dims_6d(id, 2,
     .                                1, 1, 1, 1, 1, 1,
     .                            mtdat, 1, 1, 1, 1, 1)

* wft
      CALL ef_set_work_array_dims_6d(id, 3,
     .                                1, 1, 1, 1, 1, 1,
     .                           mtwork, 1, 1, 1, 1, 1)

* ts
      CALL ef_set_work_array_dims_6d(id, 4,
     .                                1, 1, 1, 1, 1, 1,
     .                           mtwork, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE fft_inverse_compute(id, arg_1, arg_2, result, 
     .                               a, b, wft, ts)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

      REAL a(wrk1lox:wrk1hix, wrk1loy:wrk1hiy, wrk1loz:wrk1hiz,
     .       wrk1lot:wrk1hit, wrk1loe:wrk1hie, wrk1lof:wrk1hif)
      REAL b(wrk2lox:wrk2hix, wrk2loy:wrk2hiy, wrk2loz:wrk2hiz,
     .       wrk2lot:wrk2hit, wrk2loe:wrk2hie, wrk2lof:wrk2hif)
      REAL wft(wrk3lox:wrk3hix, wrk3loy:wrk3hiy, wrk3loz:wrk3hiz,
     .         wrk3lot:wrk3hit, wrk3loe:wrk3hie, wrk3lof:wrk3hif)
      REAL ts(wrk4lox:wrk4hix, wrk4loy:wrk4hiy, wrk4loz:wrk4hiz,
     .        wrk4lot:wrk4hit, wrk4loe:wrk4hie, wrk4lof:wrk4hif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable''s axes. 

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER arg, ntime, nfreq
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      CHARACTER*128 err_msg

* In addition to the array wft, will allocate an integer array iwft.
* Previously a portion of the work array wft was used for this, sending 
* a real array into fftpack routines that want an integer array.


      INTEGER iwft(15)

   40 FORMAT ('FFT_INVERSE encountered missing data at (i,j,k,l,m,n)',
     .        6I5)
   50 FORMAT ('FFT_INVERSE encountered missing data at L=', I5)

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  number of freq points.

      arg = 1
      nfreq = (arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) + 1)
      ntime = nfreq* 2

*  Check that arg1 and arg2 have the same axes.

      IF ( arg_lo_ss(X_AXIS,ARG1) .NE. arg_lo_ss(X_AXIS,ARG2) ) THEN
         err_msg = 'X Axes of A and B coefficients must agree'
         GOTO 999
      ENDIF
      IF ( arg_lo_ss(Y_AXIS,ARG1) .NE. arg_lo_ss(Y_AXIS,ARG2)  ) THEN
         err_msg = 'Y Axes of A and B coefficients must agree'
         GOTO 999
      ENDIF
      IF ( arg_lo_ss(Z_AXIS,ARG1) .NE. arg_lo_ss(Z_AXIS,ARG2) ) THEN
         err_msg = 'Z Axes of A and B coefficients must agree'
         GOTO 999
      ENDIF
      IF ( arg_lo_ss(T_AXIS,ARG1) .NE. arg_lo_ss(T_AXIS,ARG2)  ) THEN
         err_msg = 'T Axes of A and B coefficients must agree'
         GOTO 999
      ENDIF
      IF ( arg_lo_ss(E_AXIS,ARG1) .NE. arg_lo_ss(E_AXIS,ARG2)  ) THEN
         err_msg = 'E Axes of A and B coefficients must agree'
         GOTO 999
      ENDIF
      IF ( arg_lo_ss(F_AXIS,ARG1) .NE. arg_lo_ss(F_AXIS,ARG2)  ) THEN
         err_msg = 'F Axes of A and B coefficients must agree'
         GOTO 999
      ENDIF

* Set WFT and IWFT for fft; prime factorization of ntime and trig functions.

      CALL rffti (ntime, wft, iwft)

      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 600 n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m1 = arg_lo_ss(E_AXIS,ARG1)
      DO 500 m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      k1 = arg_lo_ss(Z_AXIS,ARG1)
      DO 300 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 200 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            i1 = arg_lo_ss(X_AXIS,ARG1)
            DO 100 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

* Calculate the inverse FFT for each series which have no missing data.
* Make a simple error message if we have a 1-D input time series.

               l1 =arg_lo_ss(T_AXIS,ARG1)
               DO 400 l = 1, nfreq

                  IF (arg_1(i1,j1,k1,l1,m1,n1) .EQ. bad_flag(ARG1) .OR.
     .                arg_2(i1,j1,k1,l1,m1,n1) .EQ. bad_flag(ARG2)) THEN
                     IF ( i1 .EQ. ef_unspecified_int4 .AND.
     .                    j1 .EQ. ef_unspecified_int4 .AND.
     .                    k1 .EQ. ef_unspecified_int4 .AND.
     .                    m1 .EQ. ef_unspecified_int4 .AND.
     .                    n1 .EQ. ef_unspecified_int4 ) THEN
                        WRITE (err_msg, 50) l
                     ELSE
                        WRITE (err_msg, 40) i1,j1,k1,l,m1,n1
                     ENDIF
                     GO TO 999 
                  ENDIF
                  A(l,1,1,1,1,1) = arg_1(i1,j1,k1,l1,m1,n1)
                  B(l,1,1,1,1,1) = arg_2(i1,j1,k1,l1,m1,n1)

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 400           CONTINUE

               CALL FFTINV (ntime, ts, a, b, wft, iwft)

               DO 410 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 
                   result(i,j,k,l,m,n) = ts(l,1,1,1,1,1)
 410           CONTINUE

               i1 = i1 + arg_incr(X_AXIS,ARG1)
 100        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200     CONTINUE

         k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300  CONTINUE

         m1 = m1 + arg_incr(E_AXIS,ARG1)
 500  CONTINUE

         n1 = n1 + arg_incr(F_AXIS,ARG1)
 600  CONTINUE

      RETURN

 999  CALL ef_bail_out (id, err_msg)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
