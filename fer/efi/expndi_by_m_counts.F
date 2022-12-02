* 
*  expndi_by_m_counts.F
* 
*  Ansley Manke (from expndi_by_z_counts)
*  4/2014
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
* 
* Expand argument 1 by creating a 2D XE structure from it with the 
* separation between X rows given by the lengths specified in 
* argument 2. Argument 2 is a list of length N, where N is the number of 
* X rows to create, and each item in the list is the length of that 
* column (the length of the cruise or profile). Thus the X rows created 
* are "ragged" Typical use is that arg 1 is a collection of vertical casts 
* appended end-to-end and that arg 2 is the number of values in each cast.
* The sum of the values in arg 2 is the length of argument 1. 

*  Args 3 and 4 are constants -- the size of the X and E axes, respectively
 
*  In this subroutine we provide information about
*  the function.  The user configurable information 
*  consists of the following:
* 
*  descr              Text description of the function
* 
*  num_args           Required number of arguments
* 
*  axis_inheritance   Type of axis for the result
*                        ( CUSTOM, IMPLIED_BY_Z_ARGS, q, ABSTRACT )
*                        CUSTOM          - user defined axis
*                        IMPLIED_BY_Z_ARGS - same axis as the incoming argument
*                        NORMAL          - the result is normal to this axis
*                        ABSTRACT        - an axis which only has index values
* 
*  piecemeal_ok       For memory optimization:
*                        axes where calculation may be performed piecemeal
*                        ( YES, NO )
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


      SUBROUTINE expndi_by_m_counts_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .  'Expand arg1 into an XE grid. (Split I variable into E-Rows)' )

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id, IMPLIED_BY_ARGS, NORMAL, 
     .     NORMAL, NORMAL, IMPLIED_BY_ARGS, NORMAL)

* Tell Ferret to run the following instead, if the arguments are strings.
      CALL ef_set_alt_fcn_name(id, 'EXPNDI_BY_M_COUNTS_STR')

      arg = 1
      CALL ef_set_arg_name(id, arg, 'Data')
      CALL ef_set_arg_desc(id, arg, '1-D variable to expand')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'Row Sizes')
      CALL ef_set_arg_desc(id, arg,
     .          '1-D list on E axis: count of values in each feature')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, YES, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'Result X axis')
      CALL ef_set_arg_desc(id, arg,
     .          'Variable on a 1-D grid, axis of length max_row_len')
      CALL ef_set_axis_influence_6d(id, arg, YES, NO, NO, NO, NO, NO)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

* 
*  In this subroutine we compute the result
* 
      SUBROUTINE expndi_by_m_counts_compute(id, arg_1, arg_2, arg_3, 
     .                               result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy,
     .	   mem1loz:mem1hiz, mem1lot:mem1hit,
     .	   mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .     mem2loz:mem2hiz, mem2lot:mem2hit,
     .	   mem2loe:mem2hie, mem2lof:mem2hif)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy,
     .     mem3loz:mem3hiz, mem3lot:mem3hit,
     .	   mem3loe:mem3hie, mem3lof:mem3hif)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .     memresloz:memreshiz, memreslot:memreshit,
     .     memresloe:memreshie, memreslof:memreshif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(6), res_hi_ss(6), res_incr(6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS), arg_hi_ss(6,EF_MAX_ARGS),
     .     arg_incr(6,EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l,m,n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER imax, iout, isum, mmax, mrow, mout

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* limits on output arrays
        imax = res_hi_ss(X_AXIS)
        mmax = res_hi_ss(E_AXIS)

* sum of counts in arg 2 must be length of x-dimension of arg1

      do i2 = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)
      do j2 = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)
      do k2 = arg_lo_ss(Z_AXIS,ARG2), arg_hi_ss(Z_AXIS,ARG2)
      do L2 = arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2)
      do n2 = arg_lo_ss(F_AXIS,ARG2), arg_hi_ss(F_AXIS,ARG2)
         isum = 0
         DO m2 = arg_lo_ss(E_AXIS,ARG2), arg_hi_ss(E_AXIS,ARG2)
            IF  (arg_2(i2,j2,k2,l2,m2,n2) .EQ. bad_flag(ARG2) .OR. 
     .           INT(arg_2(i2,j2,k2,l2,m2,n2)) .LT. 0)
     .          CALL EF_BAIL_OUT(id, 'arg 2 must be non-missing counts GE zero')
            isum = isum + INT(arg_2(i2,j2,k2,l2,m2,n2))
         ENDDO
         IF (isum .NE. (arg_hi_ss(X_AXIS,ARG1)-arg_lo_ss(X_AXIS,ARG1)+1) )
     .       CALL EF_BAIL_OUT(id, 
     .         'arg 2 values (counts) must sum to x-dimension length of of arg1 ')
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      ENDDO

* pre-fill the entire output result with missing value flags
*  (since length of each profile may have different number of pts)
      result = bad_flag_result


* fill in the points as specified by the counts

      n1 = arg_lo_ss(F_AXIS,ARG1)
      n2 = arg_lo_ss(F_AXIS,ARG2)
      DO 400 n=res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      l1 = arg_lo_ss(T_AXIS,ARG1)
      l2 = arg_lo_ss(T_AXIS,ARG2)
      DO 400 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

      k1 = arg_lo_ss(Z_AXIS,ARG1)
      k2 = arg_lo_ss(Z_AXIS,ARG2)
      DO 400 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      DO 400 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

         i1 = arg_lo_ss(X_AXIS,ARG1)
         i2 = arg_lo_ss(X_AXIS,ARG2)

         m1 = arg_lo_ss(E_AXIS,ARG1)
         m2 = arg_lo_ss(E_AXIS,ARG2)
 
            i1 = 1
            DO 200 mout=arg_lo_ss(E_AXIS,ARG2),arg_hi_ss(E_AXIS,ARG2)
               DO 100 iout = 1, NINT(arg_2(i2,j2,k2,l2,mout,n1))
                 IF ( arg_1(i1,j1,k1,l1,m1,n1) .NE. bad_flag(1) )
     .             result(iout,j,k,l,mout,n) = arg_1(i1,j1,k1,l1,m1,n1)

                  i1 = i1 + 1
 100           CONTINUE
 200        CONTINUE

         j1 = j1 + arg_incr(Y_AXIS,ARG1)
         j2 = j2 + arg_incr(Y_AXIS,ARG2)
         k1 = k1 + arg_incr(Z_AXIS,ARG1)
         k2 = k2 + arg_incr(Z_AXIS,ARG2)
         l1 = l1 + arg_incr(T_AXIS,ARG1)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         n1 = n1 + arg_incr(F_AXIS,ARG1)
         n2 = n2 + arg_incr(F_AXIS,ARG2)
 400  CONTINUE
      
         
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

