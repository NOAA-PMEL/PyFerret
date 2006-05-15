*
* taxmonth.F
*
* Jing Y. Li 
* May 4th 2006
*
* This function returns month specified by the first argument (variable 
* containing time values) from the second argument (variable from which time 
* encoding will be inferred).
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


      SUBROUTINE tax_month_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Returns months of time axis coordinate values' )
   
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
      SUBROUTINE tax_month_work_size(id)

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
      SUBROUTINE tax_month_compute(id, arg_1, arg_2, result, taxdat)

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
      INTEGER iyear, imon, day_of_mon
      CHARACTER*3 cmon
      INTEGER numtimes
      CHARACTER*3 months(12)

      INTEGER i,j,k,l,m 
      INTEGER i2, j2, k2, l2

      DATA months/'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
     .            'AUG', 'SEP', 'OCT', 'NOV', 'DEC'/

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*     Get number of time of variable 2 
      numtimes = 
     .     (arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1)/
     .      arg_incr(T_AXIS,ARG2)

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
     .             taxdat(1,1,1,arg_1(i,j,k,l)),1,datebuf)

*     If datebuf is in form "DD-MON-YEAR HH:MM:SS". Read date and time.
                READ (datebuf,110,err=900) day_of_mon, cmon, iyear

                DO 50 m=1,12
                  IF (cmon .EQ. months(m)) imon=m
 50             CONTINUE  

                result(i,j,k,l) = imon 

                l2 = l2 + arg_incr(T_AXIS,ARG2)
 100          CONTINUE

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
     .  'Error assigning dates/times to timestamp for tax_year',  
     .  datebuf

      RETURN
      END
