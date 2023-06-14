*
* fc_isubset.F
*
* Ansley Manke 11/28/2011
* See ticket 1897
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*
* This function returns a subset of data given start indices and
* sizes of features in a 1D array.
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


      SUBROUTINE fc_isubset_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 'Returns a subset of a feature collection' )

      CALL ef_set_num_args(id, 4)
      CALL ef_set_axis_inheritance(id, CUSTOM, NORMAL, NORMAL, NORMAL)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'index_list')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 
     .     'starting indices of features of interest')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'lengths')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 
     .     'lengths of the features of interest')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'nx')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 
     .      'length of result (sum of lengths in arg 2)')
      CALL ef_set_arg_unit(id, arg, ' ')

      arg = 4
      CALL ef_set_arg_name(id, arg, 'FullData')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 
     .     'input data: 1D list from which to select features')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

      SUBROUTINE fc_isubset_result_limits(id)

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


      SUBROUTINE fc_isubset_custom_axes(id)

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

      CALL ef_set_custom_axis (id, X_AXIS, 1., val, 1., 'X', NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE fc_isubset_compute(id, arg_1, arg_2, arg_3, arg_4, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

	  INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .           mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .           mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, 
     .           mem4loz:mem4hiz, mem4lot:mem4hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

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

      INTEGER i,j,k,l
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2
      INTEGER i4, j4, k4, l4
      INTEGER m, n, count
      CHARACTER*100 errtxt

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


* check that the lists are 1D

      errtxt = 'Argument 1 must be a 1D list'
      n = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1
      DO i = Y_AXIS, T_AXIS
         m = arg_hi_ss(i,ARG1) - arg_lo_ss(i,ARG1) + 1 
         IF (n.GT.1 .AND. m.GT.1) GOTO 9000
         n = MAX(n,m)
      ENDDO

      errtxt = 'Argument 2 must be a 1D list'
      n = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      DO i = Y_AXIS, T_AXIS
         m = arg_hi_ss(i,ARG2) - arg_lo_ss(i,ARG2) + 1
         IF (n.GT.1 .AND. m.GT.1) GOTO 9000
         n = MAX(n,m)
      ENDDO
      
* Put the requested features into the result

      i2=arg_lo_ss(X_AXIS,ARG2)
      j2=arg_lo_ss(Y_AXIS,ARG2)
      k2=arg_lo_ss(Z_AXIS,ARG2)
      l2=arg_lo_ss(T_AXIS,ARG2)

      i = res_lo_ss(X_AXIS)

      DO i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
      DO j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
      DO k1 = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
      DO l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)

         IF ( arg_1(i1,j1,k1,l1) .EQ. bad_flag(1) ) GOTO 600  ! done
         IF ( arg_2(i2,j2,k2,l2) .EQ. bad_flag(2) ) GOTO 600

	 i4 = INT(arg_1(i1,j1,k1,l1))     ! start index
         count = INT(arg_2(i2,j2,k2,l2))  ! number to put

         l4 = arg_lo_ss(T_AXIS,ARG4)
         DO l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
	    k4 = arg_lo_ss(Z_AXIS,ARG2)
            DO k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               j4 = res_lo_ss(Y_AXIS)
	       DO j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

                  DO n = 1, count
                     result(i,j,k,l) = arg_4(i4,j4,k4,l4)
                     i4 = i4 + 1
                     i = i + 1
                  ENDDO
 
                  j4 = j4 + arg_incr(Y_AXIS,ARG4)
               ENDDO
               k4 = k4 + arg_incr(Z_AXIS,ARG4)
            ENDDO

            l4 = l4 + arg_incr(T_AXIS,ARG4)
        ENDDO

      i2 = i2 + arg_incr(X_AXIS,ARG2)
      j2 = j2 + arg_incr(Y_AXIS,ARG2)
      k2 = k2 + arg_incr(Z_AXIS,ARG2)
      l2 = l2 + arg_incr(T_AXIS,ARG2)

      ENDDO
      ENDDO
      ENDDO
      ENDDO

 600  CONTINUE
      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
 9000 CALL EF_BAIL_OUT(id, errtxt)
      END
