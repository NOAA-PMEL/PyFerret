*
* tax_jday.F
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


      SUBROUTINE tax_jday_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Returns days of year of time axis coordinate values' )
   
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 1)

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
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE tax_jday_work_size(id)

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
      INTEGER array_num, nt1, nt2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)


*  Allocate double the length of the T axis of ARG2 for REAL*8 work array.

      nt1 = arg_lo_ss(T_AXIS,ARG2)

      nt2 = nt1 + 
     .     2* (arg_hi_ss(T_AXIS,ARG2)  - arg_lo_ss(T_AXIS,ARG2) + 1)

      array_num = 1

      CALL ef_set_work_array_dims(id, array_num,
     .     nt1, 1, 1, 1, nt2, 1, 1, 1)


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE tax_jday_compute(id, arg_1, arg_2, result, taxdat) 

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .           mem2loz:mem2hiz, mem2lot:mem2hit)

      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

*  Declare work array

      REAL*8 taxdat(wrk1lox:wrk1lox+(wrk1hix-wrk1lox)/2,wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)

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
      INTEGER L2, LL
      LOGICAL TM_FPEQ, first
      CHARACTER*255 err_msg
      REAL   rdate, rdate_last
      REAL*8 ddate, d1, d2
      INTEGER i,j,k,l,m 
      INTEGER i1, j1, k1, l1

*  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4), tmodulo

*  Check to see if time axis of arg 2 is modulo

      CALL ef_get_axis_info(id, 2, ax_name, ax_units, backward,
     .     modulo, regular)
      tmodulo = modulo(4)


      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_coordinates(id, ARG2, T_AXIS,
     .   arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2), taxdat)


      imon = 0
      first = .TRUE.
      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

           k1 = arg_lo_ss(Z_AXIS,ARG1)
           DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

              l1 = arg_lo_ss(T_AXIS,ARG1)
              DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                 rdate =  arg_1(i1,j1,k1,l1)  
                 ddate = rdate  
		 IF (rdate .NE. bad_flag(ARG1)) THEN

                    IF (.NOT.tmodulo) THEN
C  Convert date back to nearest double precision coordinate value

                       IF (.NOT.first .AND. TM_FPEQ(rdate, rdate_last)) THEN
C                        Create a message saying arg1 values at index are duplicate
		          CALL TAX_FORMAT_MESSAGE (L1, err_msg)
		          GOTO 999
		       ENDIF

		       first = .FALSE.
		       rdate_last = rdate

                       ddate = rdate
		       LL = arg_lo_ss(T_AXIS,ARG2)
		       L2 = arg_lo_ss(T_AXIS,ARG2)+1
		       d1 = ABS(taxdat(L2,1,1,1) - taxdat(LL,1,1,1))
		       LL = 1
		       DO 110 L2 = arg_lo_ss(T_AXIS,ARG2)+1, arg_hi_ss(T_AXIS,ARG2)
		          d2 = ABS(ddate - taxdat(L2,1,1,1))
		          IF (d2 .LT. d1) LL = L2
  110                  CONTINUE
		       ddate = taxdat(LL,1,1,1)
                    ENDIF  ! tmodulo

*     Get the string-date for each time
                    CALL ef_get_axis_dates(id,ARG2,ddate,1,datebuf)

*     If datebuf is in form "DD-MON-YEAR HH:MM:SS". Read date.
                    IF (.NOT. tmodulo) THEN
                       READ (datebuf,1010,err=900) day_of_mon, cmon, iyear

*     modulo: Datebuf is in form "DD-MON HH:MM:SS". Read date.
                    ELSE
                       READ (datebuf,1010,err=900) day_of_mon, cmon
                       iyear = 1901  ! nominally non-leap year 
                    ENDIF

                    CALL JULIAN_DAY_OF_YEAR(cmon, day_of_mon, iyear, 
     .                day_of_year, days_this_year)
                    result(i,j,k,l) = day_of_year 

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

 1010  FORMAT (i2, 1x, a3, 1x, i4) 

      RETURN 
  900 CONTINUE
      WRITE (err_msg,*) 
     .  'Error assigning dates/times to timestamp for tax_jday',  
     .  datebuf
     
      RETURN 
 999  CALL ef_bail_out (id, err_msg)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************


      RETURN
      END
