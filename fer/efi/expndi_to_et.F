* 
*  expndi_to_et.F
* 
*  Ansley Manke
*  8/31/2016
* DSGTODO  Make use of dsg-handling upgrade
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* 
* Expand argument 1 to a 2D E-T variable, using argument 2, a RowSize variable to
* separate the time series, and placing each time series at the correct
* location on the time series defined by argument 3. 
* Typical use is that arg 1 is a collection of time series in a Timeseries DSG
* file, and a common time series variable has been defined for all of the data.
 
*  In this subroutine we provide information about
*  the function.  The user configurable information 
*  consists of the following:
* 
*  descr              Text description of the function
* 
*  num_args           Required number of arguments
* 
*  axis_inheritance   Type of axis for the result
*                        ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                        CUSTOM          - user defined axis
*                        IMPLIED_BY_ARGS - same axis as the incoming argument
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


      SUBROUTINE expndi_to_et_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .  'Expand arg1 into a T-E grid defined by Rowsize and Tseries' )

      CALL ef_set_num_args(id, 4)
      CALL ef_set_axis_inheritance_6d(id,
     .                                NORMAL, NORMAL,
     .                                NORMAL, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, NORMAL)
      CALL ef_set_num_work_arrays(id, 2)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'var data')
      CALL ef_set_arg_desc(id, arg, 'Variable to expand, on I axis')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'time data')
      CALL ef_set_arg_desc(id, arg, 
     .          'Timesteps for variable to expand, on same I axis')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'Rowsize (E direction)')
      CALL ef_set_arg_desc(id, arg,
     .          'Length of each individual time series')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, YES, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'Result T')
      CALL ef_set_arg_desc(id, arg,
     .          'A variable on the desired result T axis')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, YES, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE expndi_to_et_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays, X/Y/Z/T/E/F dimensions
*
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER mth, mtl

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

     
* 
      mtl = arg_lo_ss(T_AXIS,ARG4)
      mth = mtl + (arg_hi_ss(T_AXIS,ARG4) - arg_lo_ss(T_AXIS,ARG4))

*  taxlo
      CALL ef_set_work_array_dims_6d(id, 1,
     .                              mtl, 1, 1, 1, 1, 1,
     .                              mth, 1, 1, 1, 1, 1)

*  taxhi
      CALL ef_set_work_array_dims_6d(id, 2,
     .                              mtl, 1, 1, 1, 1, 1,
     .                              mth, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


* 
*  In this subroutine we compute the result
* 
      SUBROUTINE expndi_to_et_compute(id, 
     .            arg_1, arg_2, arg_3, arg_4,
     .            result, taxlo, taxhi)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id
      
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz,
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz,
     .           mem3lot:mem3hit, mem3loe:mem3hie, mem3lof:mem3hif)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .           mem4lot:mem4hit, mem4loe:mem4hie, mem4lof:mem4hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

     
      REAL*8 taxlo(wrk1lox:wrk1hix)
      REAL*8 taxhi(wrk2lox:wrk2hix)

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

      INTEGER i, j, k, l, m, n, i1, k1
      INTEGER index1, index2, lpos
      REAL    sum, count, t1, t2

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      k1 = arg_lo_ss (Z_AXIS,ARG1)  ! unspecified_int4


* make sure arguments 1 and 2 on same grid
      IF ( (arg_lo_ss(X_AXIS,ARG1) .NE. arg_lo_ss(X_AXIS,ARG2))
     . .OR.(arg_hi_ss(X_AXIS,ARG1) .NE. arg_hi_ss(X_AXIS,ARG2)) ) CALL
     .           EF_BAIL_OUT(id, 'X limits of args 1 and 2 mismatched')

      do i = 2,6
         IF ( arg_hi_ss(i,ARG1) - arg_lo_ss(i,ARG1) + 1  .NE. 1
     .    .OR.arg_hi_ss(i,ARG2) - arg_lo_ss(i,ARG2) + 1  .NE. 1 ) CALL
     .     EF_BAIL_OUT(id, 'arguments 1 and 2 must be on X axis only')
      enddo

* make sure RowSize is valid

      sum = 0.
      DO m = arg_lo_ss(E_AXIS,ARG3), arg_hi_ss(E_AXIS,ARG3)
         sum = sum + arg_3(k1,k1,k1,k1,m,k1)
      enddo

      IF (sum .NE. arg_hi_ss(X_axis,ARG1) - arg_lo_ss(X_axis,ARG1) + 1 ) 
     .  CALL EF_BAIL_OUT(id, 'Values of arg 2 must sum to the length of arg 1')

* Get output-axis time coordinates

      CALL ef_get_box_limits(id, ARG4, T_AXIS,
     .   arg_lo_ss(T_AXIS, ARG4), arg_hi_ss(T_AXIS, ARG4), taxlo, taxhi)

* pre-fill the entire output result with missing value flags
*  (since each t-series in the input has different start and end times)

      result = bad_flag_result

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      n = res_lo_ss(F_AXIS)

* fill in the points 
      

      index1 = 1
      m = res_lo_ss(E_AXIS)
      DO m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
         index2 = index1 + INT(arg_3(k1,k1,k1,k1,m,k1)) - 1
	 t1 = arg_2(index1,k1,k1,k1,k1,k1)
	 t2 = arg_2(index2,k1,k1,k1,k1,k1)
	 DO l = arg_lo_ss(T_AXIS,ARG4), arg_hi_ss(T_AXIS,ARG4)
	    IF (t1 .GE. taxlo(l) .AND. t1 .LT. taxhi(l) ) EXIT
	 ENDDO
         lpos = l

	 if (index2 .GT. res_hi_ss(T_AXIS)) index2 = res_hi_ss(T_AXIS)
	 sum = 0.0
	 count = 0.
	 DO i1 = index1, index2
	    IF (arg_1(i1,k1,k1,k1,k1,k1) .NE. bad_flag(ARG1) ) THEN
	       count = count + 1.
	       sum = sum + arg_1(i1,k1,k1,k1,k1,k1)
	    ENDIF
	 ENDDO

	 IF (count .GT. 0.) THEN 
	    result(i,j,k,lpos,m,n) = sum/count
	 ELSE
	    result(i,j,k,lpos,m,n) = bad_flag_result
	 ENDIF

	 index1 = index2+1

      ENDDO
         


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

