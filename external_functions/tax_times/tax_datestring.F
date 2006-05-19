*
* tax_datestring.F
*
* Jing Y. Li 
* May 2006
* 18-may-2006 ACM change to compare only the first 3 characters of the precision string.
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
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, tax_datestring )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )


      SUBROUTINE tax_datestring_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,'Demo Function: tests string utilities' )

      CALL ef_set_num_args(id, 3)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, STRING_RETURN)

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

* In this subroutine we compute the result
*
      SUBROUTINE tax_datestring_compute(id, arg_1, arg_2, arg_3, 
     .           result)

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
      INTEGER rtype, slen
      CHARACTER*20 datestring
      CHARACTER*3 precistring
      REAL*8 ddate

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
     .     i3,j3,k3,l3,slen,precistring)

      CALL LOWER(precistring)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)


         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                 ddate =  arg_1(i1,j1,k1,l1)

C     Get the string value of dates and put them in the result.
                 CALL EF_GET_AXIS_DATES 
     .                (id, ARG2, ddate, 1, datestring)

                 If(precistring .EQ. 'sec') THEN
                     slen = 20
                     CALL EF_PUT_STRING (datestring, slen,
     .                    result(1,i,j,k,l))
                 
                 ELSE If(precistring .EQ. 'min') THEN
                     slen = 17
                     CALL EF_PUT_STRING (datestring(1:17), slen,
     .                    result(1,i,j,k,l))

                 ELSE If(precistring .EQ. 'hou') THEN
                     slen = 14
                     CALL EF_PUT_STRING (datestring(1:14), slen,
     .                    result(1,i,j,k,l))

                 ELSE If(precistring .EQ. 'day') THEN
                     slen = 11
                     CALL EF_PUT_STRING (datestring(1:11), slen, 
     .                    result(1,i,j,k,l))

                 ELSE IF(precistring .EQ. 'mon') THEN
                     slen = 8
                     CALL EF_PUT_STRING (datestring(4:11), slen,
     .                    result(1,i,j,k,l))

                 ELSE IF(precistring .EQ. 'yea') THEN
                     slen = 4
                     CALL EF_PUT_STRING (datestring(8:11), slen,
     .                    result(1,i,j,k,l))

                 ELSE
                     slen = 20
                     CALL EF_PUT_STRING (datestring, slen,
     .                    result(1,i,j,k,l))

                 ENDIF
                 
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

