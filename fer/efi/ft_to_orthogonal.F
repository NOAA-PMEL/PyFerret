*
* ft_to_orthogonal.F
*
* Ansley Manke 2/2015
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
* 
*
* Given a variable from a Forecast-Time dataset, and
* the 2D time variable and a time axis representing all
* the times in the set, return the variable in orthogonal
* form: properly put onto the new time axis.
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
*                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ft_to_orthogonal )
*                       CUSTOM          - user defined axis
*                       IMPLIED_BY_ARGS - same axis as the incoming argument
*                       NORMAL          - the result is normal to this axis
*                       ABSTRACT        - an axis which only has index values
*
* piecemeal_ok       For memory optimization:
*                       axes where calculation may be performed piecemeal
*                       ( YES, NO )


      SUBROUTINE ft_to_orthogonal_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************


      CALL ef_set_desc(id,
     .        'Convert data with 2D Time coordinate to orthogonal FT grid' )
      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 1)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'VAR')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'Variable with 2D Time coordinate')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'TIME2D')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'The 2-Dimensional Time coordinate variable')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'ALLTIME')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'A variable on the output time axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, YES, NO, NO)

      RETURN 
      END

***********************************************************************

* Request an amount of storage to be supplied by Ferret and passed
* as additional arguments.

      SUBROUTINE ft_to_orthogonal_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

*
* Set the work array X/Y/Z/T/E/F dimensions
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)

      INTEGER ntout, nt2
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

* Coordinate data from argument 3

      ntout = 1 + arg_hi_ss(T_AXIS,ARG3) - arg_lo_ss(T_AXIS,ARG3)

* output t coordinates
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                            ntout, 1, 1, 1, 1, 1)

      RETURN
      END

* **********************************************************************

* In this subroutine we compute the result
*
      SUBROUTINE ft_to_orthogonal_compute(id, arg_1, arg_2, arg_3, 
     .                                    result, taxout)

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

      REAL result( memreslox:memreshix, memresloy:memreshiy, 
     .             memresloz:memreshiz, memreslot:memreshit,
     .             memresloe:memreshie, memreslof:memreshif)
     
      REAL taxout(wrk1lox:wrk1hix)

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

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER l_off, del_l, del_n
      LOGICAL its_dsg
      CHARACTER*255 err_msg
      REAL time2d_ft, timeout
       
* Does the variable come from a Discrete Sampling Geometries dataset?  

      CALL EF_GET_ITS_DSG(id, ARG1, its_dsg )
      IF (its_dsg) THEN
         err_msg = 
     . 'Data is from a Discrete Sampling Geometries (DSG) dataset. Function does not apply.'
       GOTO 999
      ENDIF

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_coordinates(id, ARG3, T_AXIS, arg_lo_ss(T_AXIS, ARG3),
     .                        arg_hi_ss(T_AXIS, ARG3), taxout)

1000  FORMAT ('2D time array mismatch with new time axis: time2d', 
     . f8.2, ' time axis ', i3, f8.2)

* Fill the result with missing-flag

      result = bad_flag_result


      i2 = arg_lo_ss(X_AXIS,ARG2)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      k2 = arg_lo_ss(Z_AXIS,ARG2)
      l2 = arg_lo_ss(T_AXIS,ARG2)
      m2 = arg_lo_ss(E_AXIS,ARG2)
      n2 = arg_lo_ss(F_AXIS,ARG2)

      i3 = arg_lo_ss(X_AXIS,ARG3)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)
      m3 = arg_lo_ss(E_AXIS,ARG3)
      n3 = arg_lo_ss(F_AXIS,ARG3)

      l_off = 0
      del_l = arg_2(i2,j2,k2,l2+1,m2,n2) - arg_2(i2,j2,k2,l2,m2,n2)
      del_n = (arg_2(i2,j2,k2,l2,m2,n2+1) - arg_2(i2,j2,k2,l2,m2,n2))/del_l

      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         DO 400 l2 = arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2)

	    l = l2 + l_off
            time2d_ft = arg_2(i2,j2,k2,l2,m2,n2)

*  These should match.  (commented out now; something wrong with the
*   declaration of work arrays with this function; Ferret crashes on
*   trying to free the memory...)

	    timeout = taxout(l)
            IF (time2d_ft .NE. timeout) THEN
	       write (err_msg, 1000) time2d_ft, l, timeout
	       GOTO 999  
            ENDIF
	   
            m1 = arg_lo_ss(E_AXIS,ARG1)
            DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            j1 = arg_lo_ss(Y_AXIS,ARG1)
            DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            i1 = arg_lo_ss(X_AXIS,ARG1)
            DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

	       result(i,j,k,l,m,n) = arg_1(i1,j1,k1,l2,m1,n1)

            i1 = i1 + arg_incr(X_AXIS,ARG1)
 100        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200        CONTINUE

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300        CONTINUE

            m1 = m1 + arg_incr(E_AXIS,ARG1)
 500        CONTINUE

 400     CONTINUE

         n2 = n2 + arg_incr(F_AXIS,ARG2)
         n1 = n1 + arg_incr(F_AXIS,ARG1)
	 l_off = l_off + del_n
 600  CONTINUE

      RETURN

 999  CALL ef_bail_out (id, err_msg)

* **********************************************************************

      END
