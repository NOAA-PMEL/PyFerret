*
* lsl_lowpass.F
*
* LSL-low_pass_filter from Ned Cokelet.
*  Ansley Manke
*  Dec 6 1999
*
* Returns the argument filtered with Least Squares Lanzcos filter in time.
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


      SUBROUTINE lsl_lowpass_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 label

      WRITE (label, 10) 
   10 FORMAT ( 'Returns Least Squares Lanzcos filter of ',
     .          'equally-spaced time series' )
      CALL ef_set_desc(id,label)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_num_work_arrays(id, 3)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_desc(id, arg, 'data to be filtered')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'cutoff_period')
      CALL ef_set_arg_desc(id, arg, 
     .   'period at which filter attains 1/2 amplitude. <=N')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'filter_span')
      CALL ef_set_arg_desc(id, arg, 
     . 'number of input data points used in each filtered output point')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE lsl_lowpass_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER mt

      INTEGER iwork
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mt = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

*  x -- input time series
      iwork = 1
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mt, 1, 1, 1)

*  y -- output time series
      iwork = 2
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mt, 1, 1, 1)

*  h -- work array for subroutine LSL_low_pass_filter

      iwork = 3
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mt, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END



*
* In this subroutine we compute the result
*
      SUBROUTINE lsl_lowpass_compute(id, arg_1, arg_2, arg_3, result, 
     .                              x, y, h)

      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .     mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .     mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .     mem3loz:mem3hiz, mem3lot:mem3hit)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER i, j, k, l, n
      INTEGER i1, j1, k1, l1
      INTEGER nt
      INTEGER filter_span
      REAL cutoff_period
      CHARACTER*255 err_msg

*  Dimension the work arrays.

      REAL x(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL y(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL h(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)

*  Get argument and result subscripts and bad-data flags.

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Get cutoff parameter.

      cutoff_period = arg_2(arg_lo_ss(X_AXIS,ARG2), 
     .            arg_lo_ss(Y_AXIS,ARG2), arg_lo_ss(Z_AXIS,ARG2), 
     .            arg_lo_ss(T_AXIS,ARG2))

*  Get time percent parameter.

      filter_span = arg_3(arg_lo_ss(X_AXIS,ARG3), 
     .            arg_lo_ss(Y_AXIS,ARG3), arg_lo_ss(Z_AXIS,ARG3), 
     .            arg_lo_ss(T_AXIS,ARG3))

*  Get N = length of time series.

      nt  =  1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

      IF (filter_span .GT. nt) THEN
         WRITE (err_msg, 10)
   10    FORMAT('LSL_FILTER: The filter span must be less than or ', 
     .             'equal to the number of points')
         GOTO 999
      ENDIF
 
      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

* Compute the filtered time series.

               n = 1
               DO 100 l1=arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
                  x(n,1,1,1) = arg_1(i1,j1,k1,l1)
                  n = n + 1
 100           CONTINUE

               CALL LSL_low_pass_filter (x, nt, cutoff_period, 
     .            bad_flag(ARG1), filter_span, h, y)

*  Put the filtred series in result

               n = 1
               DO 110 l= res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
                  IF (y(n,1,1,1) .NE. bad_flag(ARG1)) THEN
                     result(i,j,k,l) = y(n,1,1,1)
                  ELSE
                     result(i,j,k,l) = bad_flag_result
                  ENDIF
                  n = n + 1
 110           CONTINUE


               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE
      
      RETURN

 999  CALL ef_bail_out (id, err_msg)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

      subroutine LSL_low_pass_filter( 
     1  x, N, cutoff_period, missing_value, filter_span, h, y )
c
c Least Squares Lanczos (LSL) low-pass filter:  This subroutine 
c low-pass filters an equally spaced time series using least-squares 
c approximation to the ideal low-pass filter of Bloomfield with Lanczos 
c convergence factors.  It is very similar to subroutine LOPASS in 
c Chapter 6, p. 149, of
c 
c Bloomfield, P., 1976, Fourier Analysis of Time Series: An 
c    Introduction, John Wiley & Sons, New York, 258 pp.
c
c The main difference is that the present routine takes account of
c missing values in the input time series.
c
c Inputs:
c
c x(N) = a real array of equally spaced points to low-pass filter
c N    = the number of points in x
c cutoff_period = the cutoff period (the period at which the filter
c                 attains 1/2 amplitude or 1/4 "energy") measured in
c                 units of delta t.  The cutoff_period must be less than
c                 or equal to N.
c filter_span = the number of input data points used in each filtered
c               output point.  A wide filter gives a narrow frequency
c               response transition band, but leads to ringing near
c               data discontinuities and loss of filtered values at the
c               end points and surrounding missing values.  A narrow
c               filter reduces ringing and output data loss, but gives
c               a wider frequency transition width, i.e. it falls off
c               less rapidly at freqs. higher than the cutoff.
c               The filter transition region lies in the period range
c               between N*cutoff_period/(N + cutoff_period) and 
c               N*cutoff_period/(N - cutoff_period). 
c missing_value = the missing value flag that signals to exclude flagged
c                 values from the computation.
c Outputs:
c
c filter_span = the filter span should be an odd integer.  It is set to 
c               the next lower odd number if the input is even.
c y(N)        = the output array of filtered values.  Values near the
c               ends and near gaps are filled with the missing value
c               flag.
c
c Note on tidal filtering:  For hourly time series containing tidal
c signals, some investigators use this filter with a 35-hour cutoff period
c and a filter span of xxx hours to remove at least 99.5 % of the energy for
c periods less than 25 hours.
c
c Adapted from Bloomfield by E. D. Cokelet, NOAA/PMEL, 3 Dec 1999    
c
c        1         2         3         4         5         6         7
c23456789012345678901234567890123456789012345678901234567890123456789012
c

c acm  explicitly define and type all variables
      integer n, i, j, ib, ie
      real*4 cutoff_period, missing_value
      real*8 pi, d, omega_c, h0, con, summ, temp, x1, x2, d1, d2

      integer filter_span, half_span
      real*4 x(*), h(*), y(*), z

c      d(z) = sin(z)/z      ! define statement function  (doesnt work on osf)

      pi = 4.*atan(1.d0)

c Make sure filter span is odd
c
      half_span = int( (filter_span - 1) / 2 )
      filter_span = 2*half_span + 1
c
c
c Calculate the filter weights and normalize them
c
      omega_c = 2 * pi / cutoff_period
c
      h0 = omega_c / pi
      con = 2.*pi/float(filter_span)
      summ = h0
      do 10 i = 1, half_span
         d1 = sin(float(i)*omega_c)/ (float(i)*omega_c)
         d2 = sin(float(i)*con)/ (float(i)*con)
c         h(i) = h0 * d( float(i)*omega_c ) * d( float(i)*con )
         h(i) = h0 * d1 * d2
         summ = summ + 2. * h(i)
 10   continue
c
      h0 = h0/summ
      do 20 i = 1, half_span
         h(i) = h(i) / summ
 20   continue
c
c
c Put missing_value in the first and last half_span elements to
c account for end effects
c
      do 30 i =1, half_span
         y(i) = missing_value
         j = N - i + 1
         y(j) = missing_value
 30   continue
c
c
c Filter by convolving.  
c To account for gaps in the time series, set to missing any filtered
c values that would require missing values to compute them.
c
      ib = half_span + 1
      ie = N - half_span
c
      do 50 i = ib, ie
         if (x(i) .eq. missing_value) then
            temp = missing_value
            go to 45
         end if
         temp = h0 * x(i)
         do 40 j = 1, half_span
   	    x1 = x(i - j)
   	    x2 = x(i + j)
   	    if ( ( x1 .eq. missing_value ) .or. 
     1          ( x2 .eq. missing_value ) ) then
   		temp = missing_value
   		go to 45
   	    end if
	    temp = temp + h(j) * ( x1 + x2 )
 40      continue
 45      continue
         y(i) = temp

 50   continue
c
      return
      end

c      real*8 function d(z)
c
cc      d(z) = sin(z)/z      ! define statement function  (doesnt work on osf)
c
c      real z
c      d = sin(z)/ z
c
c      return
c      end
