*
* dates.F
*
*
* This function tests ef_get_axis_dates

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


      SUBROUTINE dates_init(id)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CALL ef_set_desc(id,
     .        'Returns a string array of dates derived ' //
     .        'from the time axis and the given offsets')
      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_result_type(id, STRING_RETURN)
      CALL ef_set_num_work_arrays(id, 1)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'Offsets')
      CALL ef_set_arg_desc(id, arg,
     .        'Offsets from the time given by time axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

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
      SUBROUTINE dates_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

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
      INTEGER array_num, nt

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nt = arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1) + 1

* The work array does not have to dimensioned along the time axis,
* but could be if it makes the code clearer.  Later we just treat it
* as a one-dimensional array, so just dimension along the first axis.

* taxdat  time axis coordinates
*         This is going to be a double precision array
*         so allocate twice the size of the time axis
*         in case REAL is REAL*4
      array_num = 1
      CALL ef_set_work_array_dims_6d(id, array_num,
     .                                1, 1, 1, 1, 1, 1,
     .                           2 * nt, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE dates_compute(id, arg_1, result, taxdat)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

* String arrays are always viewed as double precision arrays
* Thus the initial dimension of one or two depending on whether
* REAL is REAL*8 or REAL*4
      INTEGER strdf
#ifdef double_p
      PARAMETER (strdf = 1)
#else
      PARAMETER (strdf = 2)
#endif

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)

      REAL result(strdf, memreslox:memreshix, memresloy:memreshiy,
     .                   memresloz:memreshiz, memreslot:memreshit,
     .                   memresloe:memreshie, memreslof:memreshif)

* Ignore the unused singleton final dimensions in the work arrays
      REAL*8 taxdat(wrk1lox:wrk1hix/2)

* After initialization, the 'res_' arrays contain indexing information
* for the result axes.  The 'arg_' arrays will contain the indexing
* information for each variable's axes.

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER i,  j,  k,  l,  m,  n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER taxnum, slen, idx
      REAL*8 newtime
      CHARACTER*20 datebuf
      CHARACTER*255 errtxt

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* Make sure the argument has a time axis
      IF ( (arg_lo_ss(T_AXIS, ARG1) .EQ. ef_unspecified_int4) .AND.
     .     (arg_hi_ss(T_AXIS, ARG1) .EQ. ef_unspecified_int4) ) THEN
         errtxt = 'Offsets does not have a time axis'
         GOTO 999
      ENDIF

* Get the time axis coordinates
      CALL ef_get_coordinates(id, ARG1, T_AXIS, arg_lo_ss(T_AXIS, ARG1),
     .                        arg_hi_ss(T_AXIS, ARG1), taxdat)

* Loop through the argument time offsets, assigning the result time strings
      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

       m1 = arg_lo_ss(E_AXIS,ARG1)
       DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

*       taxdat indices start at 1 (given in dates_work_size)
        taxnum = 1
        l1 = arg_lo_ss(T_AXIS,ARG1)
        DO 400 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         k1 = arg_lo_ss(Z_AXIS,ARG1)
         DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

          j1 = arg_lo_ss(Y_AXIS,ARG1)
          DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

           i1 = arg_lo_ss(X_AXIS,ARG1)
           DO 100 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            IF ( arg_1(i1,j1,k1,l1,m1,n1) .NE. bad_flag(ARG1) ) THEN

               newtime = taxdat(taxnum) + arg_1(i1,j1,k1,l1,m1,n1)

*              Convert newtime into a date string.
*              ef_get_axis_dates can work with an array of times,
*              returning an array of date strings; here we just do one
               CALL ef_get_axis_dates(id, ARG1, newtime, 1, datebuf)

*              Get the actual length of the string
               DO 15 idx = 20, 1, -1
                  IF ( datebuf(idx:idx) .NE. ' ' ) THEN
                     slen = idx
                     GOTO 25
                  ENDIF
  15           CONTINUE

*              Empty string if we get here
               slen = 0

  25           CONTINUE

            ELSE

*              Use an empty string as the undefined value
               datebuf = ' '
               slen = 0

            ENDIF

*           Put the date string (or empty string) into the string array
            CALL ef_put_string(datebuf, slen, result(1,i,j,k,l,m,n))

            i1 = i1 + arg_incr(X_AXIS,ARG1)
 100       CONTINUE

           j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200      CONTINUE

          k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300     CONTINUE

         taxnum = taxnum + 1
         l1 = l1 + arg_incr(T_AXIS,ARG1)
 400    CONTINUE

        m1 = m1 + arg_incr(E_AXIS,ARG1)
 500   CONTINUE

       n1 = n1 + arg_incr(F_AXIS,ARG1)
 600  CONTINUE

      RETURN

 999  CALL EF_BAIL_OUT(id, errtxt)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END
