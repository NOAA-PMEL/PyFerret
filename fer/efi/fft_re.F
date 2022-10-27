*------------------------------fft_re.F--------------------------------
*
* fft_re.F
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
* Dec 1998
*
* v5.50  *acm* 10/02 Simplify the error message for missing data when the 
*                    input is a simple 1-D time series.
* V5.53  *acm*  6/03 Correct the frequency axis:  why was there a factor of
*                     1.001 in freqn = 1.001*yquist?
* V7.65 *acm*  10/22 Fix for issue 126: incorrect frequency axis
*
* This function returns the real part of the FFT transform. The result time
* axis is a custom axis in period 1./time  The time axis must be regular.

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


      SUBROUTINE fft_re_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 'Computes real part of fft transform' )
      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, CUSTOM,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'Variable with regular time axis.')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, NO, YES, YES)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END


      SUBROUTINE fft_re_custom_axes(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                      
*
*     Use utility functions to get grid information about the 
*     argument; use this info to determine the custom axis lo/hi/del 

      INTEGER nfreq_lo_l, nfreq_hi_l
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER arg, nd, i
      CHARACTER outunits*32
      REAL boxsize(1), yquist, freq1, freqn, half
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), modulo(6), regular(6)

      arg = 1
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      CALL ef_get_axis_info_6d(id, arg, ax_name, ax_units,
     .                         backward, modulo, regular)

      CALL ef_get_box_size(id, arg, T_AXIS, arg_lo_ss(T_AXIS,arg), 
     .                     arg_lo_ss(T_AXIS,arg), boxsize)

      IF ( arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) .EQ. 0 ) THEN
         nfreq_lo_l = 1
         nfreq_hi_l = 2
      ELSE
         nfreq_lo_l = arg_lo_ss(T_AXIS,arg)
         nfreq_hi_l = arg_hi_ss(T_AXIS,arg)
      ENDIF

      nd = abs(nfreq_hi_l - nfreq_lo_l) + 1

C  The FFT code returns frequencies W(k) for k=1 to N/2, with N/2 rounded down

C Fix for issue 126: ef_set_freq_axis was incorrect and is not needed
C now that real numbers are double precision
C The FFT code returns frequencies W(k) for k=1 to N/2, with N/2 rounded down
C The fundamental frequency freq1 based on 1/N with no rounding.

      half = FLOAT(nd)/2.
      yquist = 1./(2.*boxsize(1))		! Nyquist frequency

      freq1 = yquist/ half
      freqn = yquist

      DO i = 1, nd
	if (float(i)* freq1 .GT. freqn) exit
      ENDDO
      IF (i .GT. half) freqn = float(i-1)* freq1

C  Set label for the frequency axis CYC/units.

      outunits = 'CYC/' // ax_units(T_AXIS)

      CALL ef_set_custom_axis (id, T_AXIS, freq1, freqn, freq1, 
     .                         outunits, NO)\

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

      SUBROUTINE fft_re_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER ntime, nfreq
      INTEGER arg
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      arg = 1
      ntime = arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) + 1

C  The FFT code returns frequencies W(k) for k=1 to N/2, with N/2 rounded down

      nfreq = ntime/ 2
      if (ntime .ne. nfreq*2) nfreq = (ntime+1)/2 ! see FFTPACK comments in rfftf

      CALL ef_set_axis_limits(id, T_AXIS, 1, nfreq)

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
      SUBROUTINE fft_re_work_size(id)

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

      mtwork = 2.5* mtdat + 15

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
     .                            mtdat, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE fft_re_compute(id, arg_1, result, a, b, wft, ts)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)

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

      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), modulo(6), regular(6), its_dsg
      INTEGER arg, nd, nf
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      CHARACTER*128 err_msg

   40 FORMAT ('FFT_RE encountered missing data at (i,j,k,l,m,n)', 6I5)
   50 FORMAT ('FFT_RE encountered missing data at L=', I5)

* Does the variable come from a Discrete Sampling Geometries dataset?  
* IF so we aren't (yet) handling its time coordinates correctly.

      CALL EF_GET_ITS_DSG(id, ARG1, its_dsg )
      IF (its_dsg) THEN
         err_msg = 
     . 'Data is from a Discrete Sampling Geometries (DSG) dataset. Function not yet implemented for DSG data.'
	GOTO 999
      ENDIF

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  number of time points.

      arg = 1
      nd = (arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) + 1)
      nf = nd/ 2

*  Check that the time axis is regular.

      CALL ef_get_axis_info_6d(id, 1, ax_name, ax_units,
     .                         backward, modulo, regular)
      IF (.NOT. regular(T_AXIS)) THEN
         WRITE (err_msg, *) 'Time axis must be a regular axis'
         GO TO 999
      ENDIF

*  Set WFT for fft; prime factorization of ND and trig functions.

      CALL rffti (nd, wft)

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

* Calculate the FFT for each time series which have no missing data.
* Make a simple error message if we have a 1-D input time series.

               l1 =arg_lo_ss(T_AXIS,ARG1)
               DO 400 l = 1, nd

                  IF (arg_1(i1,j1,k1,l1,m1,n1) .EQ. bad_flag(ARG1)) THEN
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
                  ts(l,1,1,1,1,1) = arg_1(i1,j1,k1,l1,m1,n1)

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 400           CONTINUE

               CALL four_re (nd, ts, a, b, wft)

               DO 410 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 
                   result(i,j,k,l,m,n) = a(l,1,1,1,1,1)
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
