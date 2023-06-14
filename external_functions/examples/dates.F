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



***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CALL ef_set_desc(id,
     .        'Demo function: Returns a string array of dates corresponding ' //
     .        'to timesteps')
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_result_type(id, STRING_RETURN)
      CALL ef_set_num_work_arrays(id, 1)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'Timesteps')
      CALL ef_set_arg_desc(id, arg,
     .        'Variable on a time axis, containing offsets from the ' //
     .        'coordinates of the axis, in the units of the axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

*     Set data type as string
      arg = 2
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      CALL ef_set_arg_name(id, arg, 'Precision')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'Output precision: full, sec, ' //
     .     'min, hour, day, month, year')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

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

      IMPLICIT NONE
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
      SUBROUTINE dates_compute(id, arg_1, arg_2, result, taxdat)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz,
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* Ignore the unused singleton Y, Z, T, E, and F dimensions in the work arrays
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

      CHARACTER*3 LOWER3

      INTEGER i,  j,  k,  l,  m,  n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER taxnum, dlen, idx, dim, prec
      REAL*8 newtime
      CHARACTER*32 precstr
      CHARACTER*3 low3str
      CHARACTER*32 datebuf
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
      dim = T_AXIS

*     Get the precision for output
      i2=arg_lo_ss(X_AXIS, ARG2)
      j2=arg_lo_ss(Y_AXIS, ARG2)
      k2=arg_lo_ss(Z_AXIS, ARG2)
      l2=arg_lo_ss(T_AXIS, ARG2)
      m2=arg_lo_ss(E_AXIS, ARG2)
      n2=arg_lo_ss(F_AXIS, ARG2)

*     Get the output presicion (string)
      CALL ef_get_string_arg_element_6d(id, ARG2, arg_2,
     .     i2,j2,k2,l2,m2,n2,dlen,precstr)
      low3str = LOWER3(precstr)

      IF (low3str .EQ. 'ful') THEN
*         'dd-MMM-yyyy HH:mm:ss.SSS' or 'yyyy-MM-dd HH:mm:ss.SSS' date format
          prec = 7
      ELSE IF (low3str .EQ. 'sec') THEN
*         'dd-MMM-yyyy HH:mm:ss' or 'yyyy-MM-dd HH:mm:ss' date format
          prec = 6
      ELSE IF (low3str .EQ. 'min') THEN
*         'dd-MMM-yyyy HH:mm' or 'yyyy-MM-dd HH:mm' date format
          prec = 5
      ELSE IF (low3str .EQ. 'hou') THEN
*         'dd-MMM-yyyy HH' or 'yyyy-MM-dd HH' date format
          prec = 4
      ELSE IF (low3str .EQ. 'day') THEN
*         'dd-MMM-yyyy' or 'yyyy-MM-dd' date format
          prec = 3
      ELSE IF (low3str .EQ. 'mon') THEN
*         'MMM-yyyy' or 'yyyy-MM' date format
          prec = 2
      ELSE IF (low3str .EQ. 'yea') THEN
*         'yyyy' date format
          prec = 1
      ELSE
*         'dd-MMM-yyyy HH:mm:ss.SSS' or 'yyyy-MM-dd HH:mm:ss.SSS' date format
          prec = 7
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
               CALL ef_get_axis_dates(id, ARG1, newtime, dim, 1, 
     .                                prec, dlen, datebuf)

            ELSE

*              Use an empty string as the undefined value
               datebuf = ' '
               dlen = 0

            ENDIF

*           Put the date string (or empty string) into the string array
            CALL ef_put_string(datebuf, dlen, result(i,j,k,l,m,n))

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

****************************************************************
*
      CHARACTER*3 FUNCTION LOWER3(TEXT)
      CHARACTER*(*) TEXT
*
*     Returns the first three characters of a string in lowercase.
*
      CHARACTER*1 LETTER
      INTEGER ISHIFT,I
      ISHIFT=ICHAR('a')-ICHAR('A')
      DO 10 I=1,3
        LETTER=TEXT(I:I)
        IF('A'.LE.LETTER.AND.LETTER.LE.'Z') THEN
          LETTER=CHAR(ICHAR(LETTER)+ISHIFT)
        END IF
        LOWER3(I:I) = LETTER
   10 CONTINUE
      RETURN
      END

