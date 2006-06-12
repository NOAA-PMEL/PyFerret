*
* tax_year.F
*
* Jing Y. Li 
* May 31 2006
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

      INCLUDE 'ferret_cmn/EF_Util.cmn'

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
* In this subroutine we compute the result
*
      SUBROUTINE tax_tstep_compute(id, arg_1, arg_2, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

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
      CHARACTER*20 datebuf
      INTEGER iyear, dayofmon, hour, minute, second
      CHARACTER*3 cmon
      
      CHARACTER*20 t0string
      INTEGER t0year, t0day
      CHARACTER*3 t0mon

      REAL*8 ddate

      CHARACTER*16 axname(4), axunits(4)
      LOGICAL bkw1(4), mod1(4), reg1(4)
      CHARACTER*16 tunit

      REAL timesincet0

      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*     Get the time origin (t0) string
      i2 = arg_lo_ss(X_AXIS, ARG2)
      j2 = arg_lo_ss(Y_AXIS, ARG2)
      k2 = arg_lo_ss(Z_AXIS, ARG2)
      l2 = arg_lo_ss(T_AXIS, ARG2)
 
      CALL ef_get_string_arg_element(id, ARG2, arg_2,
     .     i2, j2, k2, l2, slen, t0string)

*     Get t0's year, month, and day
      READ (t0string, 110, err=800) t0day, t0mon, t0year

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

                ddate = arg_1(i1,j1,k1,l1)

*     Get the string-date for each time
                CALL ef_get_axis_dates(id,ARG1,ddate,1,datebuf)

                print *, datebuf
*     If datebuf is in form "DD-MON-YEAR HH:MM:SS". Read date.
*                READ (datebuf,110,err=900) dayofmon, cmon, iyear
*                READ (datebuf,120,err=900) hour, minute, second

                tunit = axunits(4)

                CALL time_since_t0(t0day, t0mon, t0year, datebuf, 
     .                             tunit, timesincet0)
 
                result(i,j,k,l) = timesincet0

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
      SUBROUTINE time_since_t0(t0day, t0mon, t0year, 
     .                         datebuf, tunit, timesincet0)
      
      INTEGER t0day, t0year, dayofmon
      CHARACTER*3 t0mon
      CHARACTER*20 datebuf
      CHARACTER*16 tunit
      REAL timesincet0
      INTEGER daysincet0

      INTEGER iyear, hour, minute, second
      CHARACTER*3 cmon

      CHARACTER*3 months(12)
      INTEGER it0mon, imon

      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
     .            'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

      print *, datebuf

*     Get current date
      READ (datebuf,110,err=900) dayofmon, cmon, iyear
      READ (datebuf,120,err=900) hour, minute, second

*     Convert a month to a number
      DO 50 m=1,12
          IF (t0mon .EQ. months(m)) it0mon=m
          IF (cmon .EQ. months(m)) imon=m
 50   CONTINUE

      print *, iyear

      CALL LOWER(tunit)

      print *, tunit

*     Compute time since t0, check time axis unit
      IF( tunit(1:3) .EQ. 'sec') THEN
          timesincet0=10.
      ELSE IF(tunit(1:3) .EQ. 'min') THEN
          timesincet0=20.
      ELSE IF(tunit(1:3) .EQ. 'hou') THEN
          timesincet0=30.

      ELSE IF(tunit(1:3) .EQ. 'day') THEN
          CALL days_since_t0(dayofmon, imon, iyear, t0day,
     .                       it0mon, t0year,daysincet0)
          timesincet0=daysincet0

      ELSE IF(tunit(1:3) .EQ. 'mon') THEN

          timesincet0= (12-it0mon) + (iyear-t0year-1)*12 + imon

      ELSE IF(tunit(1:3) .EQ. 'yea') THEN

          timesincet0 = iyear-t0year

      ELSE 
          timesincet0=0.
      ENDIF

 110  FORMAT (i2, 1x, a3, 1x, i4)
 120  FORMAT (11x,3(1x,i2))

      RETURN

 900  CONTINUE

      WRITE (err_msg,*)
     .  'Error assigning dates/times in tax_step ',
     .  datebuf

      RETURN
      END

*******************************************************************
*     Compute the days since t0
*     t0 may be before or after the current year
*
      SUBROUTINE days_since_t0(dayofmon, imon, iyear, t0day, 
     .                         it0mon, t0year, daysincet0)

      INTEGER i, j
      INTEGER yrstart,yrend
      LOGICAL isaftert0
      INTEGER dayofmon, imon, iyear
      INTEGER it0mon, t0day, t0year
      INTEGER dayofyear,daysincet0, ndaymo(12)

      DATA ndaymo/31,28,31,30,31,30,31,31,30,31,30,31/

      yrstart = t0year
      yrend = iyear-1
      isaftert0 = .TRUE.
      daysincet0 = -1

      IF(iyear .LT. t0year) THEN
          yrstart = iyear+1
          yrend = t0year-1
          isaftert0 = .FALSE.
          daysincet0 = 0
      ENDIF

*     Total days between yrstart and yrend
      DO 200 i = yrstart, yrend

*     Check if it is a leap year
          IF ( ((mod(i,4) .EQ. 0) .AND. (mod(i,100) .NE. 0))
     .     .OR.
     .     (mod(i, 400) .EQ. 0)) THEN
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
     .     .OR.
     .     (mod(iyear, 400) .EQ. 0)) THEN
          ndaymo(2) = 29
      ELSE
          ndaymo(2) = 28
      ENDIF

*     Compute day of current year
      dayofyear = dayofmon
      DO 300 j =1, imon -1
              dayofyear = dayofyear + ndaymo(j)
 300      CONTINUE

      IF(isaftert0) THEN
*     For year after t0, add days of current year 
          daysincet0 = daysincet0 + dayofyear
      ELSE
*     For year before t0, subtract days of current year
          daysincet0 = -1*(daysincet0 +
     .                     ndaymo(2) + 7*31 + 4*30- dayofyear+1)
      ENDIF

*     *****************************
*     Adjust days in the year of t0
*     *****************************
      IF ( ((mod(t0year,4) .EQ. 0) .AND. (mod(t0year,100) .NE. 0))
     .     .OR.
     .     (mod(t0year, 400) .EQ. 0)) THEN
          ndaymo(2) = 29
      ELSE
          ndaymo(2) = 28
      ENDIF

*     Compute day of current year
      dayofyear = t0day
      DO 400 j =1, it0mon -1
              dayofyear = dayofyear + ndaymo(j)
 400      CONTINUE

      IF(isaftert0) THEN
*     For year after t0, subtract days of year t0
          daysincet0 = daysincet0 - dayofyear
      ELSE
*     For year before t0, add days of year t0
          daysincet0 = -1*(daysincet0 +
     .                     ndaymo(2) + 7*31 + 4*30 + dayofyear + 1)
      ENDIF

      RETURN
      END

****************************************************************
*
      SUBROUTINE LOWER(TEXT)
      CHARACTER*(*) TEXT
*
*     Converts a string to lowercase.
*
      EXTERNAL LENGTH
      INTEGER  LENGTH
      CHARACTER*1 LETTER
      INTEGER ISHIFT,I
      ISHIFT=ICHAR('a')-ICHAR('A')
      DO 10 I=1,LENGTH(TEXT)
        LETTER=TEXT(I:I)
        IF('A'.LE.LETTER.AND.LETTER.LE.'Z') THEN
          TEXT(I:I)=CHAR(ICHAR(LETTER)+ISHIFT)
        END IF
   10 CONTINUE
      RETURN
      END

***************************************************************
*
      INTEGER FUNCTION LENGTH(TEXT)
      CHARACTER*(*) TEXT
*
*     Computes the length of a string without trailing blanks.
*
      INTEGER I

      DO 1 I=LEN(TEXT),1,-1
        IF(TEXT(I:I).NE.' ') THEN
          GO TO 2
        END IF
    1 CONTINUE
      I=1
    2 CONTINUE
      LENGTH=I

      RETURN
      END

