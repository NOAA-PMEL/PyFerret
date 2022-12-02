*
* expnd_by_len.F
*
* Ansley Manke 12/2011
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*
* This function takes a variable and a set of lengths
* and returns the variable with element 1 repeated len1 times,
* then element 2 repeated len2 times, etc.
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


      SUBROUTINE expnd_by_len_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 
     . 'Returns a variable expanded by the lengths given in arg 2' )
      CALL ef_set_num_args(id, 3)
      CALL ef_set_axis_inheritance_6d(id, ABSTRACT, NORMAL, 
     .     NORMAL, NORMAL, NORMAL, NORMAL)

* Tell Ferret to run the following instead, if the arguments are strings.
      CALL ef_set_alt_fcn_name(id, 'EXPND_BY_LEN_STR')

      arg = 1
      CALL ef_set_arg_name(id, arg, 'var')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg,'1D variable to expand')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'len')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 
     .     'lengths to expand each element of argument 1')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'nx')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 
     .      'length of result (sum of lengths in arg 2)')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

      SUBROUTINE expnd_by_len_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

* Get the size of the output axis from arg 3

      INTEGER nx, iarg
      REAL val

      iarg = 3
      CALL ef_get_one_val(id, iarg, val)
      nx = INT(val)

      call ef_set_axis_limits(id, X_AXIS, 1, nx)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE expnd_by_len_compute(id, arg_1, arg_2, arg_3, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit, 
     .           mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .           mem2loz:mem2hiz, mem2lot:mem2hit, 
     .           mem2loe:mem2hie, mem2lof:mem2hif)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .           mem3loz:mem3hiz, mem3lot:mem3hit, 
     .           mem3loe:mem3hie, mem3lof:mem3hif)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshiz, memreslof:memreshif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(6), res_hi_ss(6), res_incr(6)
      INTEGER arg_lo_ss(6,1:EF_MAX_ARGS), arg_hi_ss(6,1:EF_MAX_ARGS),
     .     arg_incr(6,1:EF_MAX_ARGS)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l,m,n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER count, indx
      CHARACTER*100 errtxt
      REAL val

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* check that the lists are 1D

      errtxt = 'Argument 1 must be a 1D list'
      n = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1
      DO i = Y_AXIS, F_AXIS
         m = arg_hi_ss(i,ARG1) - arg_lo_ss(i,ARG1) + 1 
         IF (n.GT.1 .AND. m.GT.1) GOTO 9000
         n = MAX(n,m)
      ENDDO

      errtxt = 'Argument 2 must be a 1D list'
      n = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      DO i = Y_AXIS, F_AXIS
         m = arg_hi_ss(i,ARG2) - arg_lo_ss(i,ARG2) + 1
         IF (n.GT.1 .AND. m.GT.1) GOTO 9000
         n = MAX(n,m)
      ENDDO
      
* Put the requested features into the result

      i2=arg_lo_ss(X_AXIS,ARG2)
      j2=arg_lo_ss(Y_AXIS,ARG2)
      k2=arg_lo_ss(Z_AXIS,ARG2)
      l2=arg_lo_ss(T_AXIS,ARG2)
      m2=arg_lo_ss(E_AXIS,ARG2)
      n2=arg_lo_ss(F_AXIS,ARG2)

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)
      m = res_lo_ss(E_AXIS)
      n = res_lo_ss(F_AXIS)

      DO n1 = arg_lo_ss(F_AXIS,ARG1), arg_hi_ss(F_AXIS,ARG1)
      DO m1 = arg_lo_ss(E_AXIS,ARG1), arg_hi_ss(E_AXIS,ARG1)
      DO l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
      DO k1 = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
      DO j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
      DO i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
         IF ( arg_2(i2,j2,k2,l2,m2,n2) .NE. bad_flag(2) ) THEN

            val = arg_1(i1,j1,k1,l1,m1,n1)
	    IF ( arg_1(i1,j1,k1,l1,m1,n1) .EQ. bad_flag(1) ) 
     .	       val = bad_flag_result

            count = INT(arg_2(i2,j2,k2,l2,m2,n2))    ! number to replicate

            DO indx = 1, count
               result(i,j,k,l,m,n) = val
               i = i + 1
            ENDDO

         ENDIF  ! arg_2 bad

      i2 = i2 + arg_incr(X_AXIS,ARG2)
      j2 = j2 + arg_incr(Y_AXIS,ARG2)
      k2 = k2 + arg_incr(Z_AXIS,ARG2)
      l2 = l2 + arg_incr(T_AXIS,ARG2)
      m2 = m2 + arg_incr(E_AXIS,ARG2)
      n2 = n2 + arg_incr(F_AXIS,ARG2)

      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
 9000 CALL EF_BAIL_OUT(id, errtxt)
      END
