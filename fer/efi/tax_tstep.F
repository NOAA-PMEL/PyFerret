*
* tax_tsetp.F
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
* Jing Y. Li 
* May 31 2006
* Finished March 24, 2009, ACM
* v695  *acm 2/15  Time axis may be an F axis or a T axis.
* v75  *acm* 3/18  Issue 1856, changes to allow for timesteps in fractional seconds
* v745 *acm* 12/18  Issue 1909, option to write dates in ISO8601 format
* v745 *acm* 12/18  Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd
*
* This function returns time step values
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

      SUBROUTINE tax_tstep_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg


      CALL ef_set_desc(id,
     .        'Returns time step values based on the new time origin')
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 1)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 
     .           'variable with reference time or forecast axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type(id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'desired time origin')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)


      RETURN 
      END


*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE tax_tstep_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id


* Set the work arrays, X/Y/Z/T/E/F dimensions
*
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER mth, mtl

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

     
* Get the direction of arg 1: F or T
      IF (arg_lo_ss(T_AXIS, ARG1) .NE. ef_unspecified_int4) THEN
         mtl = arg_lo_ss(T_AXIS,ARG1)
         mth = mtl + 
     .         (arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1))
      ENDIF
      IF (arg_lo_ss(F_AXIS, ARG1) .NE. ef_unspecified_int4) THEN
         mtl = arg_lo_ss(F_AXIS,ARG1)
         mth = mtl + 
     .         (arg_hi_ss(F_AXIS,ARG1) - arg_lo_ss(F_AXIS,ARG1))
      ENDIF


*  taxdat
      CALL ef_set_work_array_dims_6d(id, 1,
     .                              mtl, 1, 1, 1, 1, 1,
     .                              mth, 1, 1, 1, 1, 1)


      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE tax_tstep_compute(id, arg_1, arg_2, result, taxdat)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'


      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .           mem2loz:mem2hiz, mem2lot:mem2hit,
     .           mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* Ignore the Y/Z/T/E/F dimensions in the work arrays since
* they are not used and Fortran is column major
      REAL*8 taxdat(wrk1lox:wrk1hix)

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


      LOGICAL bkw1(6), mod1(6), reg1(6)
      INTEGER STR_SAME
      INTEGER slen, dir, get_prec, dlen
      INTEGER i, j, k, l, m, n
      INTEGER l1, n1, cal, status, len
      LOGICAL its_dsg
      REAL*8 ddate, timesincet0
      CHARACTER*16 axname(6), axunits(6)
      CHARACTER*16 tunit
      CHARACTER*30 datebuf, instring, t0string
      CHARACTER*128 err_msg


* Does the variable come from a Discrete Sampling Geometries dataset?  

      CALL EF_GET_ITS_DSG(id, ARG1, its_dsg )
      IF (its_dsg) THEN
         err_msg = 
     . 'Data is from a Discrete Sampling Geometries (DSG) dataset. Function does not apply.'
       GOTO 999
      ENDIF

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* return date string to precision of seconds
      get_prec = 6

*     Get the new time origin (t0) string
      i = arg_lo_ss(X_AXIS, ARG2)
      j = arg_lo_ss(Y_AXIS, ARG2)
      k = arg_lo_ss(Z_AXIS, ARG2)
      l = arg_lo_ss(T_AXIS, ARG2)
      m = arg_lo_ss(E_AXIS, ARG2)
      n = arg_lo_ss(F_AXIS, ARG2)
      CALL ef_get_string_arg_element_6d(id, ARG2, arg_2,
     .                       i, j, k, l, m, n, slen, instring)

* Allow them to input the date w/o a leading 0 so that either
* 1-jan-2000 or 01-jan-0000 is ok

      IF ( instring(2:2) .EQ. '-' ) THEN
         WRITE (t0string, 20) instring
  20     FORMAT ('0', a19)
      ELSE
         t0string = instring
      ENDIF
      IF (t0string(5:5) .EQ. '-' ) THEN
	 CALL EF_CONVERT_DATE_STRING(t0string, status)
	 IF (status .NE. 0) GOTO 900
      ENDIF

* Get the direction of arg 1: F or T
      IF (arg_lo_ss(T_AXIS, ARG1) .NE. ef_unspecified_int4) THEN
         CALL ef_get_coordinates(id, ARG1, T_AXIS,
     .        arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), taxdat)
         dir = T_AXIS
      ENDIF
        
      IF (arg_lo_ss(F_AXIS, ARG1) .NE. ef_unspecified_int4) THEN
         CALL ef_get_coordinates(id, ARG1, F_AXIS,
     .        arg_lo_ss(F_AXIS, ARG1), arg_hi_ss(F_AXIS, ARG1), taxdat)
         dir = F_AXIS
      ENDIF

*     Get time axis unit
      CALL ef_get_axis_info_6d(id, ARG1, axname, axunits,
     .                         bkw1, mod1, reg1)
      tunit = axunits(dir)

      IF (dir .EQ. T_AXIS) THEN
         l1 = arg_lo_ss(T_AXIS,ARG1)
         DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

*      Get the time axis value for this step
            ddate = taxdat(l1)
*      Get the date string corresponding to time ddate
            CALL ef_get_axis_dates(id, ARG1, ddate, dir, 1, get_prec, dlen, datebuf)
*      Convert to time from the new origin
            CALL time_since_t0(t0string, datebuf, tunit, timesincet0, err_msg)

*      Assign this time from the new origin to the results
            DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
            DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            result(i,j,k,l,m,n) = timesincet0

 100        CONTINUE
 200        CONTINUE
 300        CONTINUE
 500        CONTINUE
 600        CONTINUE

            l1 = l1 + arg_incr(T_AXIS,ARG1)
 400     CONTINUE

      ELSE IF (dir .EQ. F_AXIS) THEN
         n1 = arg_lo_ss(F_AXIS,ARG1)
         DO 650 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

*      Get the F axis value for this step
            ddate = taxdat(n1)
*      Get the date string corresponding to time ddate
            CALL ef_get_axis_dates(id, ARG1, ddate, dir, 1, get_prec, dlen, datebuf)
*      Convert to time from the new origin
            CALL time_since_t0(t0string, datebuf, tunit, timesincet0, err_msg)

*      Assign this time from the new origin to the results
            DO 550 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO 450 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            DO 350 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
            DO 250 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            DO 150 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

               result(i,j,k,l,m,n) = timesincet0

 150        CONTINUE
 250        CONTINUE
 350        CONTINUE
 450        CONTINUE
 550        CONTINUE

            n1 = n1 + arg_incr(F_AXIS,ARG1)
 650     CONTINUE
      ENDIF

      RETURN
 900  WRITE (err_msg,*) 
     .  'Invalid time origin given in argument 2: ', t0string
 999  CALL ef_bail_out (id, err_msg )
      ReTURN


      END


*********************************************************************
*
*     Compute the time since desired t0
*
*********************************************************************
      SUBROUTINE time_since_t0(t0string, datebuf, tunit, timesincet0, err_msg)
      
      CHARACTER*(*) t0string, datebuf, tunit, err_msg
      REAL timesincet0

      INTEGER secondsincet0, minutesincet0, hoursincet0
      INTEGER daysincet0, monthsincet0, yearsincet0

      CALL LOWER_CASE(tunit)

*     Compute time since t0, check time axis unit
      IF( tunit(1:3) .EQ. 'sec') THEN
          CALL second_since_t0(t0string, datebuf, secondsincet0, err_msg)
          timesincet0 = secondsincet0
      ELSE IF(tunit(1:3) .EQ. 'min') THEN
          CALL minute_since_t0(t0string, datebuf, minutesincet0, err_msg)
          timesincet0 = minutesincet0
      ELSE IF(tunit(1:3) .EQ. 'hou') THEN
          CALL hour_since_t0(t0string, datebuf, hoursincet0, err_msg)
          timesincet0 = hoursincet0
      ELSE IF(tunit(1:3) .EQ. 'day') THEN
          CALL day_since_t0(t0string, datebuf, daysincet0, err_msg)
          timesincet0 = daysincet0
      ELSE IF(tunit(1:3) .EQ. 'mon') THEN
          CALL month_since_t0(t0string, datebuf, monthsincet0, err_msg)
          timesincet0 = monthsincet0
      ELSE IF(tunit(1:3) .EQ. 'yea') THEN
          CALL year_since_t0(t0string, datebuf, yearsincet0, err_msg)
          timesincet0 = yearsincet0
      ELSE 
          timesincet0=0.
      ENDIF

      RETURN
      END

*******************************************************************
*     Compute second since t0
*******************************************************************
      SUBROUTINE second_since_t0(t0string, datebuf, secondsincet0, err_msg)

      CHARACTER*(*) t0string, datebuf, err_msg
      INTEGER minutesincet0, secondsincet0

      INTEGER isecond, t0second, t0hour

      t0hour = 0

      CALL minute_since_t0(t0string, datebuf, minutesincet0, err_msg)
      secondsincet0 = 60*minutesincet0

*     Get t0's minute
      READ (t0string, 110, err=800) t0second

*     Get current minute
      READ (datebuf, 110, err=900) isecond

 110  FORMAT (18x, i2)

*     Adjust minutes of current date
      secondsincet0 = secondsincet0 - (60-isecond)

*     Adjust minutes of t0
      secondsincet0 = secondsincet0 + (60 - t0second)

      RETURN
*-------------
 800  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for t0 date in tax_tstep',
     .  datebuf
      RETURN

*-------------
 900  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for current date in tax_tstep',
     .  datebuf
      RETURN

      END


*******************************************************************
*     Compute minutes since t0
*******************************************************************
      SUBROUTINE minute_since_t0(t0string, datebuf, minutesincet0, err_msg)

      CHARACTER*(*) t0string, datebuf, err_msg

      INTEGER minutesincet0, hoursincet0

      INTEGER iminute, t0minute, t0hour

      t0hour = 0

      CALL hour_since_t0(t0string, datebuf, hoursincet0, err_msg)
      minutesincet0 = 60*hoursincet0

*     Get t0's minute
      READ (t0string, 110, err=800) t0minute

*     Get current minute
      READ (datebuf, 110, err=900) iminute

 110  FORMAT (15x, i2, 3x)

*     Adjust minutes of current date
      minutesincet0 = minutesincet0 - (60-iminute)

*     Adjust minutes of t0
      minutesincet0 = minutesincet0 + (60 - t0minute)

      RETURN
*-------------
 800  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for t0 date in tax_tstep',
     .  datebuf
      RETURN

*-------------
 900  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for current date in tax_tstep',
     .  datebuf
      RETURN

      END

*******************************************************************
*     Compute hours since t0
*******************************************************************
      SUBROUTINE hour_since_t0(t0string, datebuf, hoursincet0, err_msg)

      CHARACTER*(*) t0string, datebuf, err_msg

      INTEGER hoursincet0, daysincet0
       
      INTEGER ihour, t0hour      

      t0hour = 0

      CALL day_since_t0(t0string, datebuf, daysincet0, err_msg)
      hoursincet0 = 24*daysincet0

*     Get t0's hour
      READ (t0string, 110, err=800) t0hour

*     Get current hour
      READ (datebuf, 110, err=900) ihour

 110  FORMAT (12x, i2, 6x)

*     Adjust hours of current date
      hoursincet0 = hoursincet0 - (24-ihour)

*     Adjust hours of t0
      hoursincet0 = hoursincet0 + (24 - t0hour)
      RETURN
*-------------
 800  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for t0 date in tax_tstep',
     .  datebuf
      RETURN

*-------------
 900  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for current date in tax_tstep',
     .  datebuf
      RETURN

      END
*******************************************************************
*     Compute the days since t0
*     Assume that t0 is before the current year
*******************************************************************
      SUBROUTINE day_since_t0(t0string, datebuf, daysincet0, err_msg)

      CHARACTER*(*) t0string, datebuf, err_msg

      INTEGER daysincet0

      INTEGER t0day, t0year
      CHARACTER*3 t0mon

      INTEGER dayofmon, iyear 
      CHARACTER*3 cmon

      CHARACTER*3 months(12)
      INTEGER it0mon, imon
      INTEGER i, j, m
      INTEGER yrstart,yrend
      INTEGER dayofyear, ndaymo(12)

      DATA ndaymo/31,28,31,30,31,30,31,31,30,31,30,31/

      DATA months/'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
     .            'aug', 'sep', 'oct', 'nov', 'dec'/

*     Get t0's year, month, and day
      READ (t0string, 110, err=800) t0day, t0mon, t0year
      
*     Get current date
      READ (datebuf, 110, err=900) dayofmon, cmon, iyear
   
 110  FORMAT (i2, 1x, a3, 1x, i4)

      CALL LOWER_CASE(t0mon)
      CALL LOWER_CASE(cmon)

*     Convert a month from string to number
      DO 50 m=1,12
          IF (t0mon .EQ. months(m)) it0mon=m
          IF (cmon .EQ. months(m)) imon=m
 50   CONTINUE

      yrstart = t0year
      yrend = iyear-1
      daysincet0 = 0

*     Total days between yrstart and yrend
      DO 200 i = yrstart, yrend

*     Check if it is a leap year
          IF ( ((mod(i,4) .EQ. 0) .AND. (mod(i,100) .NE. 0))
     .       .OR.
     .       (mod(i, 400) .EQ. 0)) THEN
                ndaymo(2) = 29
          ELSE
                ndaymo(2) = 28
          ENDIF

          daysincet0 = daysincet0 + (ndaymo(2) + 7*31 + 4*30)

 200  CONTINUE

*     ***************************
*     Adjust days of current year
*     ***************************
      IF ( ((mod(iyear,4) .EQ. 0) .AND. (mod(iyear,100) .NE. 0))
     .       .OR.
     .       (mod(iyear, 400) .EQ. 0)) THEN
          ndaymo(2) = 29
      ELSE
          ndaymo(2) = 28
      ENDIF

*     Compute day of current year
      dayofyear = dayofmon
      DO 300 j =1, imon -1
          dayofyear = dayofyear + ndaymo(j)
 300  CONTINUE

*     Add days of current year 
          daysincet0 = daysincet0 + dayofyear

*     *****************************
*     Adjust days in the year of t0
*     *****************************
      IF ( ((mod(t0year,4) .EQ. 0) .AND. (mod(t0year,100) .NE. 0))
     .       .OR.
     .       (mod(t0year, 400) .EQ. 0)) THEN
          ndaymo(2) = 29
      ELSE
          ndaymo(2) = 28
      ENDIF

*     Compute day of current year
      dayofyear = t0day
      DO 400 j =1, it0mon -1
          dayofyear = dayofyear + ndaymo(j)
 400  CONTINUE

*     Subtract days of year t0
      daysincet0 = daysincet0 - dayofyear

      RETURN

*-------------
 800  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for timr origin in tax_tstep',
     .  t0string
      RETURN

*-------------
 900  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for current date in tax_tstep',
     .  datebuf
      RETURN

      END

*******************************************************************
*     Compute month since t0
*******************************************************************
      SUBROUTINE month_since_t0(t0string, datebuf, monthsincet0, err_msg)

      CHARACTER*(*) t0string, datebuf, err_msg
      INTEGER monthsincet0
      CHARACTER*3 t0mon, cmon
      CHARACTER*3 months(12)
      INTEGER iyear, t0year, it0mon, imon
      INTEGER m

      DATA months/'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
     .            'aug', 'sep', 'oct', 'nov', 'dec'/

*     Get t0's year, month, and day
      READ (t0string, 110, err=800) t0mon, t0year
*     Get current date
      READ (datebuf, 110, err=900) cmon, iyear

 110  FORMAT (3x, a3, 1x, i4)

      CALL LOWER_CASE(t0mon)
      CALL LOWER_CASE(cmon)
*     Convert a month from string to number
      DO 50 m=1,12
          IF (t0mon .EQ. months(m)) it0mon=m
          IF (cmon .EQ. months(m)) imon=m
 50   CONTINUE

      monthsincet0= (12-it0mon) + (iyear-t0year-1)*12 + imon

      RETURN
*-------------
 800  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for time origin in tax_tstep: ',
     .  t0string
      RETURN
*-------------
 900  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for current date in tax_tstep: ',
     .  datebuf
      RETURN

      END

*******************************************************************
*     Compute year since t0
*******************************************************************
      SUBROUTINE year_since_t0(t0string, datebuf, yearsincet0, err_msg)

      CHARACTER*(*) t0string, datebuf, err_msg
      INTEGER yearsincet0
      INTEGER iyear, t0year

*     Get t0's year, month, and day
      READ (t0string, 110, err=800) t0year
*     Get current date
      READ (datebuf, 110, err=900) iyear
 110  FORMAT (7x, i4)

      yearsincet0 = iyear - t0year 

      RETURN
*-------------
 800  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for time origin in tax_tstep: ',
     .  t0string
      RETURN
*-------------
 900  CONTINUE
      WRITE (err_msg,*)
     .  'Error assigning dates/times for current date in tax_tstep: ',
     .  datebuf
      RETURN

      END


