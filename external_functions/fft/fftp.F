*
* fftp.F
*
* Ansley Manke
* Dec 1998
*
*
* This function computes fft phase spectrum for each time series.  Result
* time axis is a custom axis in period 1./time  The input variable must have
* the time axis specified explicitly in the function call e.g. 
*    LET sstp = fftp(ssttim[l=1:492])
* and the time axis must be regular.

*  Note current limitations as of 28-Apr-1999
*   - result T-axis units is set to FREQ: want to get the time step units and
*     set the frequency axis to 1./days, etc.
*   - Will have a utility to get the length of the input time axis at the
*     point when the custom freq axis is set up, to use in computing the
*     length of the frequency axis.  Currently need to specify explicitly, 
*     in the function call e.g. fftp(sst[l=1:400]).

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


      SUBROUTINE fftp_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CHARACTER*110 arg_desc
      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CALL ef_set_desc(id,'Computes fft phase spectra  v5.1' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, CUSTOM)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 5)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      WRITE (arg_desc, 10) 
   10 FORMAT ('Variable with regular time axis. Specify time ', 
     .         'explicitly e.g. fftp(var[l=1,120])')

      CALL ef_set_arg_desc(id, arg, arg_desc)

      CALL ef_set_axis_influence(id, arg, YES, YES, YES, NO)


*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END



      SUBROUTINE fftp_custom_axes(id)

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
      CHARACTER outunits*20

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)


      arg = 1
      IF (arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) .EQ. 0 ) THEN
         nfreq_lo_l = 1
         nfreq_hi_l = 2		! use: size of t axis from get_arg_axis_limits
      ELSE
         nfreq_lo_l = arg_lo_ss(T_AXIS,arg)
         nfreq_hi_l = arg_hi_ss(T_AXIS,arg)
      ENDIF

      arg = 1
      nd = (arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) + 1)

      nfreq = nd/ 2

*  FFT routine works on an even number of time points.

      IF (nd .NE. nfreq* 2) nd = nd-1
      nfreq = nd/ 2

      yquist = 0.5              ! Nyquist frequency
      freq1 = 1.* yquist/ float(nfreq)
      freqn = yquist

      outunits = 'FREQ'
      CALL ef_set_custom_axis (id, T_AXIS, freq1, freqn, freq1, 
     .   outunits, NO)

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

      SUBROUTINE fftp_result_limits(id)

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
      nfreq = ntime/ 2

*  FFT routine works on an even number of time points.

      IF (ntime .NE. nfreq*2) ntime = ntime-1
      nfreq = ntime/ 2

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
      SUBROUTINE fftp_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_lens(id, array #, X len, Y len, Z len, T len)
*
      INTEGER mtdat, mtwork
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mtdat = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

      mtwork = 2.5* mtdat + 15
  
* creal
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, mtdat, 1, 1, 1)

* cimag
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, mtdat, 1, 1, 1)

* wft
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, mtwork, 1, 1, 1)

* ts
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, mtdat, 1, 1, 1)

* ws
      CALL ef_set_work_array_dims (id, 5, 1, 1, 1, 1, mtdat, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE fftp_compute(id, arg_1, result, creal, cimag, wft, 
     .                ts, ws)

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
      COMMON /STOR/ mtdat, mtwork
      INTEGER mtdat, mtwork

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4)

*  Dimenstion work arrays

      REAL creal(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL cimag(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL wft(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL ts(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)
      REAL ws(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .               wrk5loz:wrk5hiz, wrk5lot:wrk5hit)

      REAL rr, ri, rad
      INTEGER nd, nf
      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1

      CHARACTER*80 err_msg

   40 FORMAT ('FFTP encountered missing data at (i,j,k,l)', 4I5)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  FFT routine works on an even number of time points.

      arg = 1
      nd = (arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) + 1)
      nf = nd/ 2
      IF (nd .NE. nf*2) nd = nd-1

*  Check that the time axis is regular.

      arg = 1
      CALL ef_get_axis_info (id, 1, ax_name, ax_units, backward,
     .     modulo, regular)

      IF (.NOT. regular(T_AXIS)) THEN
            WRITE (err_msg, *) 'Time axis must be a regular axis'
            GO TO 999
      ENDIF


*  Set WFT for fft; prime factorization of ND and trig functions.
*  Set weights for 10%, 5%  or 1% cosine window.  
*  Or CALL PROLATE (WS, ND, IP) w/ IP=1 OR 4  for pi or 4pi prolate window.)

      CALL rffti (nd, wft)
      CALL coswindow (ws, nd, 5)
      rad = 180.0/ 3.141592654

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*
* Calculate the FFT phase for each time series which have no missing data.
*
               l1 =arg_lo_ss(T_AXIS,ARG1)
               DO 100 l = 1, nd
                  IF (arg_1(i1,j1,k1,l1) .EQ. bad_flag(ARG1)) THEN
                     WRITE (err_msg, 40) i1,j1,k1,l
                     GO TO 999 
                  ENDIF

                  ts(l,1,1,1) = ws(l,1,1,1)* arg_1(i1,j1,k1,l1)
                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 100           CONTINUE


               l1 = arg_lo_ss(T_AXIS,ARG1)
               l = res_lo_ss(T_AXIS)

               CALL FOUR_RE (nd, ts, creal, cimag, wft)

               DO 110 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 
                   rr = creal(l,1,1,1)
                   ri = cimag(l,1,1,1)
                   result(i,j,k,l) = rad* ATAN2(rr, ri)
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
