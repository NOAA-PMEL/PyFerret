*
* tax_tsetp.F
*
* Jing Y. Li 
* May 31 2006
* Finished March 24, 2009, ACM
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

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Returns time step values based on the new time origin' )
   
      CALL ef_set_num_args(id, 2)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 1)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'variable with reference time axis')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_type(id, arg, STRING_ARG)

      CALL ef_set_arg_desc(id, arg, 'desired time origin')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

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
      SUBROUTINE tax_tstep_work_size(id)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_lens(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER mth, mtl

      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

*  Allocate double the length of the axes for REAL*8 work array.

      mtl = arg_lo_ss(T_AXIS,ARG1)

      mth = mtl + 
     .     2* (arg_hi_ss(T_AXIS,ARG1)  - arg_lo_ss(T_AXIS,ARG1) + 1)

*  taxdat
      CALL ef_set_work_array_dims (id, 1, mtl, 1, 1, 1, mth, 1, 1, 1)


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE tax_tstep_compute(id, arg_1, arg_2, result, taxdat)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(2, mem2lox:mem2hix, mem2loy:mem2hiy,
     .           mem2loz:mem2hiz, mem2lot:mem2hit)

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
      INTEGER STR_SAME
      CHARACTER*20 datebuf

      INTEGER iyear, dayofmon, hour, minute, second
      CHARACTER*3 cmon
      CHARACTER*80 err_msg
      
      CHARACTER*20 instring,t0string
      INTEGER t0year, t0day, slen
      CHARACTER*3 t0mon

      REAL*8 ddate

      CHARACTER*16 axname(4), axunits(4)
      LOGICAL bkw1(4), mod1(4), reg1(4)
      CHARACTER*16 tunit

      REAL timesincet0

      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2

*  Declare work array

      REAL*8 taxdat(wrk1lox:wrk1lox+(wrk1hix-wrk1lox)/2,wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_coordinates(id, ARG1, T_AXIS,
     .   arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), taxdat)

*     Get the new time origin (t0) string
      i2 = arg_lo_ss(X_AXIS, ARG2)
      j2 = arg_lo_ss(Y_AXIS, ARG2)
      k2 = arg_lo_ss(Z_AXIS, ARG2)
      l2 = arg_lo_ss(T_AXIS, ARG2)
 
*     Get the desired time origin
      CALL ef_get_string_arg_element(id, ARG2, arg_2,
     .     i2, j2, k2, l2, slen, instring)

* Allow them to input the date w/o a leading 0 so that either
* 1-jan-2000 or 01-jan-0000 is ok

      IF ( STR_SAME(instring(2:2), '-') .EQ. 0) THEN
         WRITE (t0string, 120) instring
 120     FORMAT ('0', a19)
      ELSE
         t0string = instring
      ENDIF

*     Get time axis unit
      CALL ef_get_axis_info(id, ARG1, axname, axunits, bkw1, mod1, reg1)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

           k1 = arg_lo_ss(Z_AXIS,ARG1)
           DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

              l1 = arg_lo_ss(T_AXIS,ARG1)
              DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                ddate = taxdat(l1,1,1,1)
                IF (ddate .NE. bad_flag(ARG1)) THEN

*     Get the string-date for each time
                   CALL ef_get_axis_dates(id,ARG1,ddate,1,datebuf)
                   tunit = axunits(4)
                   CALL time_since_t0(t0string,datebuf,tunit,timesincet0)
 
                   result(i,j,k,l) = timesincet0

                ELSE
                   result(i,j,k,l) = bad_flag_result
                ENDIF

                l1 = l1 + arg_incr(T_AXIS,ARG1)
 100          CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE

 110  FORMAT (i2, 1x, a3, 1x, i4)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 

 800  CONTINUE
      WRITE (err_msg,*)
     .  'Error for reading t0string in tax_tstep ',
     .  t0string

      RETURN

      END

*********************************************************************
*
*     Compute the time since desired t0
*
*********************************************************************
      SUBROUTINE time_since_t0(t0string, datebuf, tunit, timesincet0)
      
      CHARACTER*20 t0string, datebuf
      CHARACTER*16 tunit
      REAL timesincet0

      INTEGER secondsincet0, minutesincet0, hoursincet0
      INTEGER daysincet0, monthsincet0, yearsincet0

      CALL LOWER_CASE(tunit)

*     Compute time since t0, check time axis unit
      IF( tunit(1:3) .EQ. 'sec') THEN
          CALL second_since_t0(t0string, datebuf, secondsincet0)
          timesincet0 = secondsincet0
      ELSE IF(tunit(1:3) .EQ. 'min') THEN
          CALL minute_since_t0(t0string, datebuf, minutesincet0)
          timesincet0 = minutesincet0
      ELSE IF(tunit(1:3) .EQ. 'hou') THEN
          CALL hour_since_t0(t0string, datebuf, hoursincet0)
          timesincet0 = hoursincet0
      ELSE IF(tunit(1:3) .EQ. 'day') THEN
          CALL day_since_t0(t0string, datebuf, daysincet0)
          timesincet0 = daysincet0
      ELSE IF(tunit(1:3) .EQ. 'mon') THEN
          CALL month_since_t0(t0string, datebuf, monthsincet0)
          timesincet0 = monthsincet0
      ELSE IF(tunit(1:3) .EQ. 'yea') THEN
          CALL year_since_t0(t0string, datebuf, yearsincet0)
          timesincet0 = yearsincet0
      ELSE 
          timesincet0=0.
      ENDIF

      RETURN
      END

*******************************************************************
*     Compute second since t0
*******************************************************************
      SUBROUTINE second_since_t0(t0string, datebuf, secondsincet0)

      CHARACTER*20 t0string, datebuf
      CHARACTER*80 err_msg
      INTEGER minutesincet0, secondsincet0

      INTEGER isecond, t0second, t0hour

      t0hour = 0

      CALL minute_since_t0(t0string, datebuf, minutesincet0)
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
      SUBROUTINE minute_since_t0(t0string, datebuf, minutesincet0)

      CHARACTER*20 t0string, datebuf
      CHARACTER*80 err_msg
      INTEGER minutesincet0, hoursincet0

      INTEGER iminute, t0minute, t0hour

      t0hour = 0

      CALL hour_since_t0(t0string, datebuf, hoursincet0)
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
      SUBROUTINE hour_since_t0(t0string, datebuf, hoursincet0)

      CHARACTER*20 t0string, datebuf
      CHARACTER*80 err_msg

      INTEGER hoursincet0, daysincet0
       
      INTEGER ihour, t0hour      

      t0hour = 0

      CALL day_since_t0(t0string, datebuf, daysincet0)
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
      SUBROUTINE day_since_t0(t0string, datebuf, daysincet0)

      CHARACTER*20 t0string, datebuf
      CHARACTER*80 err_msg

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
      SUBROUTINE month_since_t0(t0string, datebuf, monthsincet0)

      CHARACTER*20 t0string, datebuf
      CHARACTER*80 err_msg
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
      SUBROUTINE year_since_t0(t0string, datebuf, yearsincet0)

      CHARACTER*20 t0string, datebuf
      CHARACTER*80 err_msg
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

