*------------------------------fft_im.F--------------------------------
* fft_im.F
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
*
*
* This function returns the imaginary part of the FFT transform. The result 
* time axis is a custom axis in period 1./time  The time axis must be regular.

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


      SUBROUTINE fft_im_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CHARACTER*110 arg_desc
c      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
*                                                                      V
      CALL ef_set_desc(id, 'Computes imaginary part of fft transform' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, CUSTOM)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 4)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      WRITE (arg_desc, 10) 
   10 FORMAT ('Variable with regular time axis.')

      CALL ef_set_arg_desc(id, arg, arg_desc)

      CALL ef_set_axis_influence(id, arg, YES, YES, YES, NO)


*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END


      SUBROUTINE fft_im_custom_axes(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                      
*
*     Use utility functions to get grid information about the 
*     argument; use this info to determine the custom axis lo/hi/del 

      INTEGER nfreq_lo_l, nfreq_hi_l

      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

      INTEGER arg
      INTEGER nfreq, nd

      REAL yquist, freq1, freqn

      CHARACTER outunits*32
      REAL boxsize(1)
         
      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4)

      arg = 1
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      CALL ef_get_axis_info(id, arg, ax_name, ax_units, backward,
     .     modulo, regular)

      CALL ef_get_box_size(id, arg, T_AXIS, arg_lo_ss(T_AXIS,arg), 
     .               arg_lo_ss(T_AXIS,arg), boxsize)
     

      IF ( arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) .EQ. 0 ) THEN
         nfreq_lo_l = 1
         nfreq_hi_l = 2
      ELSE
         nfreq_lo_l = arg_lo_ss(T_AXIS,arg)
         nfreq_hi_l = arg_hi_ss(T_AXIS,arg)
      ENDIF

      nd = abs(nfreq_hi_l - nfreq_lo_l) + 1


C  The FFT code returns frequencies W(k) for k=1 to N/2, with N/2 rounded down

c      nfreq = nd/2	
c      yquist = 1./(2.*boxsize(1))		! Nyquist frequency
c
c      freq1 = 1.* yquist/ float(nfreq)
c!      freqn = 1.001*yquist  !  WHY THE 1.001???
c      freqn = yquist
c
cC  Set label for the frequency axis CYC/units.
c
c      outunits = 'CYC/' // ax_units(T_AXIS)
c
c      CALL ef_set_custom_axis (id, T_AXIS, freq1, freqn, freq1, 
c     .                         outunits, NO)


C Use ef_set_freq_axis (if we could let freq1, freqn be double precision
C in the ef_set_custom_axis call, that would be equivalent.)
C
C  Set label for the frequency axis CYC/units.

      outunits = 'CYC/' // ax_units(T_AXIS)

      CALL ef_set_freq_axis (id, T_AXIS, nd, boxsize(1), outunits, NO)

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

      SUBROUTINE fft_im_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER ntime, nfreq

      INTEGER arg
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


*     Use utility functions to get context information about the argument.

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nfreq = 1

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
      SUBROUTINE fft_im_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER mtdat, mtwork
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mtdat = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

      mtwork = 2.5* mtdat + 15
  
* a
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, mtdat, 1, 1, 1)

* b
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, mtdat, 1, 1, 1)

* wft
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, mtwork, 1, 1, 1)

* ts
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, mtdat, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE fft_im_compute(id, arg_1, result, a, b, wft, ts)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id, arg

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4)

*  Dimension work arrays

      REAL a(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL b(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL wft(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL ts(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)

      INTEGER nd, nf
      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1

      CHARACTER*80 err_msg

   40 FORMAT ('FFTA encountered missing data at (i,j,k,l)', 4I5)
   50 FORMAT ('FFTA encountered missing data at L=', I5)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  number of time points.

      arg = 1
      nd = (arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) + 1)
      nf = nd/ 2

*  Check that the time axis is regular.

      CALL ef_get_axis_info (id, 1, ax_name, ax_units, backward,
     .     modulo, regular)

      IF (.NOT. regular(T_AXIS)) THEN
            WRITE (err_msg, *) 'Time axis must be a regular axis'
            GO TO 999
      ENDIF

*  Set WFT for fft; prime factorization of ND and trig functions.

      CALL rffti (nd, wft)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*
* Calculate the FFT for each time series which have no missing data.
*
               l1 =arg_lo_ss(T_AXIS,ARG1)
               DO 100 l = 1, nd
                  IF (arg_1(i1,j1,k1,l1) .EQ. bad_flag(ARG1)) THEN

                     WRITE (err_msg, 40) i1,j1,k1,l

                     IF (i1 .EQ. ef_unspecified_int4 .AND.
     .                   j1 .EQ. ef_unspecified_int4 .AND.
     .                   k1 .EQ. ef_unspecified_int4 )
     .                     WRITE (err_msg, 50) l

                     GO TO 999 
                  ENDIF

                  ts(l,1,1,1) = arg_1(i1,j1,k1,l1)
                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 100           CONTINUE


               l1 = arg_lo_ss(T_AXIS,ARG1)
               l = res_lo_ss(T_AXIS)

               CALL four_re (nd, ts, a, b, wft)

               DO 110 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 
                   result(i,j,k,l) = b(l,1,1,1)
 110           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE

      RETURN

 999  CALL ef_bail_out (id, err_msg)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
      
