*
* tax_index_datestring.F
*
*
* Returns string of dates 
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
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, tax_index_datestring )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )


      SUBROUTINE tax_index_datestring_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Returns date string for time axis coordinate values' )

      CALL ef_set_num_args(id, 3)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, STRING_RETURN)

      CALL ef_set_num_work_arrays(id, 1)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'indices of time steps to convert')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)


      arg = 2
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'variable with reference time axis')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

*     Set data type as string
      arg = 3
      CALL ef_set_arg_type (id, arg, STRING_ARG)

      CALL ef_set_arg_name(id, arg, 'C')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'output precision')
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
      SUBROUTINE tax_index_datestring_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER ntout, nt2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      ntout = 1 + arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2)

      nt2 = (ntout+1)* 2

* tax  output t axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nt2, 1, 1, 1)

      RETURN
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE tax_index_datestring_compute(id, arg_1, arg_2, arg_3, 
     .           result, tax)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy,
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .           mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(2,mem3lox:mem3hix, mem3loy:mem3hiy, 
     .     mem3loz:mem3hiz, mem3lot:mem3hit)

      REAL result(2,memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)


* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l
      INTEGER i1, j1, k1, l1
      INTEGER i3, j3, k3, l3
      INTEGER rtype, iindex, prec, dlen
      CHARACTER*32 datestring
      CHARACTER*3 precistring
      REAL*8 ddate

*  Dimension the work arrays

      REAL*8 tax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_result_type(id, rtype)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*     Get the precision for output
      i3=arg_lo_ss(X_AXIS, ARG3)
      j3=arg_lo_ss(Y_AXIS, ARG3)
      k3=arg_lo_ss(Z_AXIS, ARG3)
      l3=arg_lo_ss(T_AXIS, ARG3)

*     Get the output presicion (string)
      CALL ef_get_string_arg_element(id, ARG3, arg_3,
     .     i3,j3,k3,l3,dlen,precistring)

      CALL LOWER(precistring)
      IF (precistring .EQ. 'sec') THEN
*         'dd-MMM-yyyy HH:mm:ss' or 'yyyy-MM-dd HH:mm:ss' date format
          prec = 6
      ELSE IF (precistring .EQ. 'min') THEN
*         'dd-MMM-yyyy HH:mm' or 'yyyy-MM-dd HH:mm' date format
          prec = 5
      ELSE IF (precistring .EQ. 'hou') THEN
*         'dd-MMM-yyyy HH' or 'yyyy-MM-dd HH' date format
          prec = 4
      ELSE IF (precistring .EQ. 'day') THEN
*         'dd-MMM-yyyy' or 'yyyy-MM-dd' date format
          prec = 3
      ELSE IF (precistring .EQ. 'mon') THEN
*         'MMM-yyyy' or 'yyyy-MM' date format
          prec = 2
      ELSE IF (precistring .EQ. 'yea') THEN
*         'yyyy' date format
          prec = 1
      ELSE
*         'dd-MMM-yyyy HH:mm:ss' or 'yyyy-MM-dd HH:mm:ss' date format
          prec = 6
      ENDIF

* Get coordinates of time axis

      call ef_get_coordinates(id, ARG2, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG2), arg_hi_ss(T_AXIS, ARG2), tax)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                 iindex = arg_1(i1,j1,k1,l1)
                 ddate =  tax(iindex,1,1,1)

C     Get the string value of dates and put them in the result.
                 CALL EF_GET_AXIS_DATES (id, ARG2, ddate, 1, prec, dlen, datestring)
                 CALL EF_PUT_STRING (datestring, dlen, result(1,i,j,k,l))

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
 100           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE
     
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

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

