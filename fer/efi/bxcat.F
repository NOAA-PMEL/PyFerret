*
* bxcat.F
*
* Ansley Manke
* Dec 2015
* 
*
* This function returns the X values of the two variables,
* concatenated into one long list on a custom simple x axis. Other 
* axes are unchanged.
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


      SUBROUTINE bxcat_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*100 descrip

      WRITE (descrip, 100)
  100 FORMAT ('Concatenates the X values of two variables into one ',
     .        'list on an indexed X axis')
      CALL ef_set_desc(id, descrip)

* Tell Ferret to run the following instead, if the arguments are strings.
      CALL ef_set_alt_fcn_name(id, 'bxcat_STR')

      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                CUSTOM,          IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'First variable to concatenate')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'Second variable to concatenate')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, YES, YES, YES, YES)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END

      
 
      SUBROUTINE bxcat_custom_axes(id)
 
      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
 
      INTEGER id
      REAL size

       INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .         arg_hi_ss(6,EF_MAX_ARGS),
     .         arg_incr (6,EF_MAX_ARGS)
     
       CHARACTER*16 axname(6), axunits(6)
       LOGICAL bkw1(6), mod1(6), reg1(6)
*
*     Use utility functions to get context information about the arguments.
*
       CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
       
*  If the variable is just a raw constant, then it's an all ABSTRACT grid
*  and the axis length is 99999999.

       CALL ef_get_axis_info_6d(id, ARG2, axname, axunits,
     .                         bkw1, mod1, reg1)
       IF (arg_hi_ss(X_AXIS,ARG1) .GE. 99999999 ) THEN  ! see abst. axis size in get_memory.F
          CALL ef_get_axis_info_6d(id, ARG1, axname, axunits,
     .                         bkw1, mod1, reg1)
          IF (axname(1)(1:8) .EQ. 'ABSTRACT') arg_hi_ss(X_AXIS,ARG1) = 1
       ENDIF
 
       IF (arg_hi_ss(X_AXIS,ARG2) .GE. 99999999 ) THEN  ! see abst. axis size in get_memory.F
          CALL ef_get_axis_info_6d(id, ARG2, axname, axunits,
     .                         bkw1, mod1, reg1)
          IF (axname(2)(1:8) .EQ. 'ABSTRACT') arg_hi_ss(X_AXIS,ARG2) = 1
       ENDIF

* output X axis size

      size = (arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1) +
     .       (arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1)

* If non-blank units, the framework calls this a LONGITUDE axis... Fix this.
      CALL ef_set_custom_axis(id, X_AXIS, 1.0, size, 1.0, 'i', NO)
 
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
 



      SUBROUTINE bxcat_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER nx

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = (arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1) +
     .     (arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1)

      CALL ef_set_axis_limits(id, X_AXIS, 1, nx)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE bxcat_compute(id, arg_1, arg_2, result)

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

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      i = res_lo_ss(X_AXIS)
      DO 100 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)

         n = res_lo_ss(F_AXIS)
         DO 600 n1 = arg_lo_ss(F_AXIS,ARG1), arg_hi_ss(F_AXIS,ARG1)

         m = res_lo_ss(E_AXIS)
         DO 500 m1 = arg_lo_ss(E_AXIS,ARG1), arg_hi_ss(E_AXIS,ARG1)

         l = res_lo_ss(T_AXIS)
         DO 400 l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)

         k = res_lo_ss(Z_AXIS)
         DO 300 k1 = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)

         j = res_lo_ss(Y_AXIS)
         DO 200 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)

            IF ( arg_1(i1,j1,k1,l1,m1,n1) .EQ. bad_flag(ARG1) ) THEN
               result(i,j,k,l,m,n) = bad_flag_result
            ELSE
               result(i,j,k,l,m,n) = arg_1(i1,j1,k1,l1,m1,n1)
            ENDIF

            j = j + res_incr(Y_AXIS)
 200     CONTINUE

            k = k + res_incr(Z_AXIS)
 300     CONTINUE

            l = l + res_incr(T_AXIS)
 400     CONTINUE

            m = m + res_incr(E_AXIS)
 500     CONTINUE

            n = n + res_incr(F_AXIS)
 600     CONTINUE

         i = i + res_incr(X_AXIS)
 100  CONTINUE

* concatenate in X - dont restart count of index I

      DO 110 i2 = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)

         n = res_lo_ss(F_AXIS)
         DO 610 n2 = arg_lo_ss(F_AXIS,ARG2), arg_hi_ss(F_AXIS,ARG2)

         m = res_lo_ss(E_AXIS)
         DO 510 m2 = arg_lo_ss(E_AXIS,ARG2), arg_hi_ss(E_AXIS,ARG2)

         l = res_lo_ss(T_AXIS)
         DO 410 l2 = arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2)

         k = res_lo_ss(Z_AXIS)
         DO 310 k2 = arg_lo_ss(Z_AXIS,ARG2), arg_hi_ss(Z_AXIS,ARG2)

         j = res_lo_ss(Y_AXIS)
         DO 210 j2 = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)

            IF ( arg_2(i2,j2,k2,l2,m2,n2) .EQ. bad_flag(ARG2) ) THEN
               result(i,j,k,l,m,n) = bad_flag_result
            ELSE
               result(i,j,k,l,m,n) = arg_2(i2,j2,k2,l2,m2,n2)
            ENDIF

            j = j + res_incr(Y_AXIS)
 210     CONTINUE

            k = k + res_incr(Z_AXIS)
 310     CONTINUE

            l = l + res_incr(T_AXIS)
 410     CONTINUE

            m = m + res_incr(E_AXIS)
 510     CONTINUE

            n = n + res_incr(F_AXIS)
 610     CONTINUE

         i = i + res_incr(X_AXIS)
 110  CONTINUE

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END
