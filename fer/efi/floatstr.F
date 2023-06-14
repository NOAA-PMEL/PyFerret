*
* floatstr.F
*
* Ansley Manke 11/10
* V695 *acm* 9/15  Allow integer formats. Truncate Floating point data, write as integer
* V697 *acm* 11/15 Fix for the zero-filled versions of these formats, so that I0 and F0.0 work
*
* Returns float converted to strings according to format given. 
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
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, floatstr )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )

      SUBROUTINE floatstr_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .'Returns float converted to strings according to format specified')

      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_result_type(id, STRING_RETURN)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg,  
     .     'Floating-point variable to convert')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

*     Set data type as string
      arg = 2
      CALL ef_set_arg_type(id, arg, STRING_ARG)
      CALL ef_set_arg_name(id, arg, 'FMT')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg,  
     . 'Fortran format, in parentheses, e.g. "(F5.1)", "(I4)", or "(I3.3)"')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

* In this subroutine we compute the result
*
      SUBROUTINE floatstr_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'


      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result( memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

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

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1, p, p2
      REAL    fval
      INTEGER TM_LENSTR1, STR_UPCASE, slen, ival, isok, dot, status
      CHARACTER*32 fstring, fformat
      CHARACTER*128 err_msg
      CHARACTER*1 nullstr


      LOGICAL do_integer, zero_fill

      nullstr = ''
      isok = 3
      zero_fill = .FALSE.
      do_integer = .FALSE.

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*     Get the Fortran Format (string)
      i = arg_lo_ss (X_AXIS, ARG2)
      j = arg_lo_ss (Y_AXIS, ARG2)
      k = arg_lo_ss (Z_AXIS, ARG2)
      l = arg_lo_ss (T_AXIS, ARG2)
      m = arg_lo_ss (E_AXIS, ARG2)
      n = arg_lo_ss (F_AXIS, ARG2)
      CALL ef_get_string_arg_element_6d(id, ARG2, arg_2,
     .                   i, j, k, l, m, n, slen, fstring)

      i = STR_UPCASE(fformat, fstring)

      do_integer = .FALSE.
      slen = TM_LENSTR1( fformat )
      dot = INDEX(fformat,'.')
      DO 30, p =1, slen-1
	 IF (fformat(p:p+1) .EQ. 'F0' .AND. p+2.LT.dot)  THEN
	    zero_fill = .TRUE.
	    fformat = fformat(1:p)//fformat(p+2:slen)
	    GOTO 40
	 ENDIF
	 IF (fformat(p:p) .EQ. 'F') GOTO 40
	 
	 IF (fformat(p:p+1).EQ.'I0' .AND. fformat(p+2:p+2).NE.')')  THEN
	    zero_fill = .TRUE.
	    do_integer = .TRUE.
	    fformat = fformat(1:p)//fformat(p+2:slen)
	    GOTO 40
	 ENDIF
	 IF (fformat(p:p) .EQ. 'I') THEN
	    do_integer = .TRUE.
            GOTO 40
	 ENDIF
  30  CONTINUE
  40  CONTINUE
      

      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 600 n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m1 = arg_lo_ss(E_AXIS,ARG1)
      DO 500 m=res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      l1 = arg_lo_ss(T_AXIS,ARG1)
      DO 400 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

      k1 = arg_lo_ss(Z_AXIS,ARG1)
      DO 300 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      DO 200 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 100 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         fval = arg_1(i1,j1,k1,l1,m1,n1)
C     Write the string using the format, and put it in the result.
C     If they have sent in an integer format, truncate the data first.

         IF ( fval .NE. bad_flag(ARG1) ) THEN
	    IF (do_integer) THEN 
	       ival = INT(fval)
               WRITE (fstring, fformat) ival
            ELSE
               WRITE (fstring, fformat) fval
	    ENDIF

            slen = TM_LENSTR1(fstring)

C adjust the field width if requested by the user (from repl_exprns.F)
           IF (zero_fill) THEN
	      DO p = 1, slen
	         IF (fstring(p:p) .EQ. ' ') fstring(p:p) = '0'
	      END DO
	   ENDIF

            CALL EF_PUT_STRING(fstring, slen, result(i,j,k,l,m,n))

         ELSE

            fstring = nullstr
            slen = 0
            CALL EF_PUT_STRING (fstring, slen, result(i,j,k,l,m,n))
         ENDIF

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 100  CONTINUE

         j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200  CONTINUE

         k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300  CONTINUE

         l1 = l1 + arg_incr(T_AXIS,ARG1)
 400  CONTINUE

         m1 = m1 + arg_incr(E_AXIS,ARG1)
 500  CONTINUE

         n1 = n1 + arg_incr(F_AXIS,ARG1)
 600  CONTINUE

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
 5000 err_msg = 'Could not translate W or ZW width value '// fformat
      CALL ef_bail_out (id, err_msg)
      RETURN

      END
