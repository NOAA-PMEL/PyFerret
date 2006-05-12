*
* tax_jday1900.F
*
* Jing Y. Li 
* May 4th 2006
*
* This function returns day of current year specified by the first argument 
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
*
* 2006-05-12 add code to handle years before 1900

      SUBROUTINE tax_jday1900_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Returns days since 1900 of time axis cooedinate values' )
   
      CALL ef_set_num_work_arrays(id,1)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'this arg is passed through')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'this arg is passed through')
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
      SUBROUTINE tax_jday1900_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work array X/Y/Z/T dimensions
*
* ef_set_work_array_lens(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER array_num, nt
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

* This is going to be a double precision array, the time axis coordinates
* so allocate 2* the size of the time axis

      array_num = 1
      nt = 2*(arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1)
      CALL ef_set_work_array_dims(id, array_num, 1,1,1,1, 1,1,1,nt) 
     
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* ********************************************************************** 
      RETURN
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE tax_jday1900_compute(id, arg_1, arg_2, result, taxdat)

*      INCLUDE 'tax.h'
      
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

      REAL*8 taxdat(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .              wrk1loz:wrk1hiz, wrk1lot:wrk1hit/2) 

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
      INTEGER iyear, imon, day_of_mon, day_since_1900
      CHARACTER*3 cmon
      INTEGER numtimes

      INTEGER i,j,k,l,m 
      INTEGER i2, j2, k2, l2

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*     Get number of time of variable 2 
      numtimes = 
     .     (arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1)/
     .     arg_incr(T_AXIS,ARG2)

*     Get time axis coordinates for second variable.
      CALL ef_get_coordinates (id, ARG2, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG2), arg_hi_ss(T_AXIS, ARG2), taxdat)

      imon = 0
      i2 = arg_lo_ss(X_AXIS,ARG2)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j2 = arg_lo_ss(Y_AXIS,ARG2)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

           k2 = arg_lo_ss(Z_AXIS,ARG2)
           DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

             l2 = arg_lo_ss(T_AXIS,ARG2)
             DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

*     Get the string-date for each time
                CALL ef_get_axis_dates(id,ARG2,
     .               taxdat(1,1,1,arg_1(i,j,k,l)),1,datebuf)

*     If datebuf is in form "DD-MON-YEAR HH:MM:SS". Read date and time.
               READ (datebuf, 110, err=900) 
     .              day_of_mon, cmon, iyear

               CALL juldate1900(cmon, day_of_mon, iyear, day_since_1900)
               result(i,j,k,l) = day_since_1900 

               l2 = l2 + arg_incr(T_AXIS,ARG2)
 100         CONTINUE

               k2 = k2 + arg_incr(Z_AXIS,ARG2)
 200        CONTINUE

            j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300     CONTINUE

         i2 = i2 + arg_incr(X_AXIS,ARG2)
 400  CONTINUE

 110  FORMAT (i2, 1x, a3, 1x, i4) 

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
  900 CONTINUE

      WRITE (err_msg,*) 
     .  'Error assigning dates/times to timestamp for tax_jday1900',  
     .  datebuf

  999 ier = 0

      RETURN
      END
      
* **********************************************************************
*     Convert day of month to day since 1900
*     Returns -1 if the year is before 1900

      SUBROUTINE juldate1900(cmon, day_of_mon, iyear, day_since_1900)

      INTEGER i, j, iyear
      INTEGER yrstart,yrend
      LOGICAL isafter1900
      CHARACTER*3 cmon
      CHARACTER*3 months(12)
      INTEGER day_of_mon, day_of_year, imon, day_since_1900, ndaymo(12)

      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
     .            'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

      DATA ndaymo/31,28,31,30,31,30,31,31,30,31,30,31/

      imon = 0
      DO 100 i=1,12
          IF(cmon .EQ. months(i)) imon = i 
 100  CONTINUE

      yrstart = 1900
      yrend = iyear-1
      isafter1900 = .TRUE. 
      day_since_1900 = -1

      IF(iyear .LT. 1900) THEN
          yrstart = iyear+1
          yrend = 1899
          isafter1900 = .FALSE.          
          day_since_1900 = 0
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
 
          day_since_1900 = day_since_1900 + 
     .                    (ndaymo(2) + 7*31 + 4*30)

 200  CONTINUE

*     Add days of current year
      IF ( ((mod(iyear,4) .EQ. 0) .AND. (mod(iyear,100) .NE. 0))
     .     .OR.
     .     (mod(iyear, 400) .EQ. 0)) THEN
          ndaymo(2) = 29
      ELSE
          ndaymo(2) = 28
      ENDIF

      IF(isafter1900) THEN
*     Year after 1900
          day_of_year = day_of_mon
          DO 300 j =1, imon -1
              day_of_year = day_of_year + ndaymo(j)
 300      CONTINUE
       
          day_since_1900 = day_since_1900 + day_of_year
      ELSE
*     Year before 1900
          day_of_year = day_of_mon
          DO 400 j =1, imon -1
              day_of_year = day_of_year + ndaymo(j)
 400      CONTINUE

          day_since_1900 = -1*(day_since_1900 + 
     .                     ndaymo(2) + 7*31 + 4*30- day_of_year+1)

      ENDIF

      RETURN
      END
*                                                                      ^
*                                                                      |
*                                                                      |
* **********************************************************************
