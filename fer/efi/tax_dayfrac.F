*
* tax_dayfrac.F
*
* Jing Y. Li 
* May 4th 2006
* 27-Feb-2013 ACM  Fix ticket 2043: remove old code for handling single-
*                  precision values coming in from argument 1
*
* This function returns day fraction ([0,1]) specified by the first argument 
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
* axis_influence     Are this argument''s axes the same as the result grid?
*                       ( YES, NO )
*
* axis_extend       How much does Ferret need to extend arg limits relative to result 
*


      SUBROUTINE tax_dayfrac_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .       'Returns fraction of day of time axis coordinate values' )
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
      CALL ef_set_arg_desc(id, arg, 'time steps to convert')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'variable with reference time axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

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
      SUBROUTINE tax_dayfrac_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work array X/Y/Z/T/E/F dimensions
*
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER array_num, nt1, nt2

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

*  Allocate double the length of the T axis of ARG2 for REAL*8 work array.
      nt1 = arg_lo_ss(T_AXIS,ARG2)
      nt2 = nt1 + 
     .        2 * (arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1)

      array_num = 1
      CALL ef_set_work_array_dims_6d(id, array_num,
     .                              nt1, 1, 1, 1, 1, 1,
     .                              nt2, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE tax_dayfrac_compute(id, arg_1, arg_2, result, taxdat) 

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* Ignore the Y/Z/T/E/F dimensions in the work arrays since
* they are not used and Fortran is column major
      REAL*8 taxdat(wrk1lox:wrk1lox+(wrk1hix-wrk1lox)/2)

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

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*20 datebuf
      INTEGER iyear, day_of_mon
      INTEGER ihour, iminute, isec
      CHARACTER*3 cmon
      INTEGER L2, LL
      LOGICAL TM_FPEQ, first
      CHARACTER*255 err_msg
      REAL*8 ddate, d1, d2
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1

*  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), modulo(6), regular(6), tmodulo

*  Check to see if time axis of arg 2 is modulo
      CALL ef_get_axis_info_6d(id, ARG2, ax_name, ax_units,
     .                         backward, modulo, regular)
      tmodulo = modulo(T_AXIS)

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_coordinates(id, ARG2, T_AXIS,
     .        arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2), taxdat)

      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m1 = arg_lo_ss(E_AXIS,ARG1)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      k1 = arg_lo_ss(Z_AXIS,ARG1)
      DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            i1 = arg_lo_ss(X_AXIS,ARG1)
            DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

              first = .TRUE.

              l1 = arg_lo_ss(T_AXIS,ARG1)
              DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                  ddate = arg_1(i1,j1,k1,l1,m1,n1)
                  IF ( ddate .NE. bad_flag(ARG1) ) THEN

*     Get the date string corresponding to time ddate for the variable ARG2
                     CALL EF_GET_AXIS_DATES(id, ARG2, ddate, 1, datebuf)

                     IF ( .NOT. tmodulo ) THEN
*                       datebuf is in form "DD-MON-YEAR HH:MM:SS". Read time.
                        READ (datebuf,420,err=900) ihour, iminute, isec
 420                    FORMAT (11x, 3(1x,i2))
                     ELSE 
*                       modulo: datebuf is in form "DD-MON HH:MM:SS". Read time.
                        READ (datebuf,430,err=900) ihour, iminute, isec
 430                    FORMAT (6x, 3(1x,i2))
                     ENDIF
                     result(i,j,k,l,m,n) = ( (isec / 60. + iminute) / 60.
     .                                        + ihour ) / 24.

                  ELSE

                     result(i,j,k,l,m,n) = bad_flag_result

                  ENDIF

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 400           CONTINUE

               i1 = i1 + arg_incr(X_AXIS,ARG1)
 100        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200     CONTINUE

         k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300  CONTINUE

         m1 = m1 + arg_incr(E_AXIS,ARG1)
 500  CONTINUE

         n1 = n1 + arg_incr(F_AXIS,ARG1)
 600  CONTINUE

      RETURN

 900  WRITE (err_msg,*)
     .  'Error assigning dates/times to timestamp for tax_dayfrac',
     .  datebuf
 999  CALL ef_bail_out (id, err_msg)
 
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END
