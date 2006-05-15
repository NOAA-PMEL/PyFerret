*
* tax_yearfrac.F
*
* Jing Y. Li 
* May 4th 2006
*
* This function returns fraction of year specified by the first argument 
* (variable containing time values) from the second argument (variable from 
* which time encoding will be inferred).
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


      SUBROUTINE tax_yearfrac_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Returns fraction of year of time axis cooedinate values' )
   
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'time steps to convert')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'variable with reference time axis')
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
      SUBROUTINE tax_yearfrac_compute(id, arg_1, arg_2, result) 
      
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
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
      INTEGER iyear, imon, day_of_mon, day_of_year, days_this_year
      CHARACTER*3 cmon

      REAL*8 ddate

      INTEGER i,j,k,l,m 
      INTEGER i1, j1, k1, l1

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      imon = 0
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
                CALL ef_get_axis_dates(id,ARG2,ddate,1,datebuf)
                print *, datebuf

*     If datebuf is in form "DD-MON-YEAR HH:MM:SS". Read date and time.
                READ (datebuf, 110, err=900) day_of_mon, cmon, iyear

                CALL juldate(cmon, day_of_mon, iyear, day_of_year,
     .               days_this_year)

                result(i,j,k,l) = day_of_year/REAL(days_this_year) 

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
  900 CONTINUE

      WRITE (err_msg,*) 
     .  'Error assigning dates/times to timestamp for tax_yearfrac',  
     .  datebuf

      RETURN
      END

************************************************************************      
*     This subroutine returns day number of current year and the total 
*     number of days of current year.
      SUBROUTINE juldate(cmon, day_of_mon, iyear, day_of_year, 
     .           days_this_year)

*     Convert day of month to day of year

      INTEGER i, iyear
      CHARACTER*3 cmon
      CHARACTER*3 months(12)
      INTEGER day_of_mon, day_of_year, imon, ndaymo(12)
      INTEGER days_this_year

      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
     .            'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

      DATA ndaymo/31,28,31,30,31,30,31,31,30,31,30,31/

      imon = 0
      DO 100 i=1,12
          IF(cmon .EQ. months(i)) imon = i 
 100  CONTINUE

*     Check if it is a leap year
      IF ( ((mod(iyear,4) .EQ. 0) .AND. (mod(iyear,100) .NE. 0)) 
     .     .OR.
     .     (mod(iyear, 400) .EQ. 0)) THEN
          ndaymo(2) = 29
      END IF

*     Compute day number within current year
      day_of_year = day_of_mon
      DO 200 i =1, imon -1
          day_of_year = day_of_year + ndaymo(i) 
 200  CONTINUE

*     Compute total days of current year
      days_this_year = 0
      DO 300 i =1, 12 
          days_this_year = days_this_year + ndaymo(i)
 300  CONTINUE

      RETURN
      END
*                                                                      ^
*                                                                      |
*                                                                      |
* **********************************************************************
