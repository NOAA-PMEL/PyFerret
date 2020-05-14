*
* box_edges function
*
* Ansley Manke
* May 2009
*
* This function returns the edges of the grid boxes for 
* a 1-D variable. 
*

      SUBROUTINE box_edges_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg


      CALL ef_set_desc(id,'Returns the cell edge locations for the axis of a 1-D variable.' )

      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT, NORMAL,
     .                                NORMAL,   NORMAL,
     .                                NORMAL,   NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'Variable')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      RETURN 
      END

*
* Set the limits for abstract axis.

      SUBROUTINE box_edges_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER i, np, npmax, numax
      CHARACTER*1 axdir(6)

      DATA axdir/'X', 'Y', 'Z', 'T', 'E', 'F'/

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      npmax = 0
      numax = 0
      DO i = X_AXIS, F_AXIS 
         np = arg_hi_ss(i,ARG1) - arg_lo_ss(i,ARG1) + 1
	 IF (np.GT.1 .AND. npmax.GT.1) GOTO 5100
	 IF (arg_hi_ss(i,ARG1) .EQ. EF_unspecified_int4) np = 0
	 npmax = MAX(np, npmax)
	 IF (arg_hi_ss(i,ARG1) .NE. EF_unspecified_int4) numax = numax + 1
      ENDDO

* No grid - i.e. argument is just a number.
      IF (npmax .EQ. 0) GOTO 5200

* No axes more than 1 long if multiple axes, issue a note

      IF (npmax.EQ.1 .AND. numax.GT.1) THEN
         DO i = X_AXIS, F_AXIS 
	    IF (arg_hi_ss(i,ARG1) .NE. EF_unspecified_int4) THEN
	       CALL WARN 
     .   ('Ambiguous argument, grid has only 1-point axes. Returning edges from '
     .   //axdir(i)//' axis')
	       EXIT ! from loop
	    ENDIF
         ENDDO
      ENDIF

* +1 for box_edges
      npmax = npmax + 1 
      CALL ef_set_axis_limits(id, X_AXIS, 1, npmax)

      RETURN 

5100  CALL EF_BAIL_OUT(id, 'Argument must be 1-dimensional')
5200  CALL EF_BAIL_OUT(id, 'Argument has no axes')

      END

*
* In this subroutine we compute the result
*
      SUBROUTINE box_edges_compute(id, arg_1, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
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

      INTEGER i, i1, i2,  j, k, l, m, n
      INTEGER idir, np
      REAL    box_hi
      
      LOGICAL its_dsg
      CHARACTER*128 err_msg
   
* Does the variable come from a Discrete Sampling Geometries dataset?  
* IF so we aren't (yet) handling its time coordinates correctly.

      CALL EF_GET_ITS_DSG(id, ARG1, its_dsg )
      IF (its_dsg) THEN
         err_msg = 
     . 'Data is from a Discrete Sampling Geometries (DSG) dataset. Function not yet implemented for DSG data.'
       GOTO 999
      ENDIF

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

! Find the axis of interest. If the result is is 1 cell in all directions
! return the edges for of the first axis found.

      idir = 0
      DO i = X_AXIS, F_AXIS 
         np = arg_hi_ss(i,ARG1) - arg_lo_ss(i,ARG1) + 1
	 IF (np.EQ.res_hi_ss(X_AXIS)-1 .AND. 
     .        arg_hi_ss(i,ARG1).NE.EF_unspecified_int4) THEN
	    idir = i
	    EXIT ! from loop
	 ENDIF
      ENDDO

* Get the low subscripts into the result and the uppermost grid cell limit as the last element

      i1 = arg_lo_ss(idir,ARG1)
      i2 = arg_hi_ss(idir,ARG1)    ! box low limits

      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)
      m = res_lo_ss(E_AXIS)
      n = res_lo_ss(F_AXIS)

* Lower limits of requested cells
      CALL ef_get_box_lo_lim(id, ARG1, idir, i1, i2, 
     .           result(res_lo_ss(X_AXIS),j,k,l,m,n) )

* Upper limit of top cell
      CALL ef_get_box_hi_lim(id, ARG1, idir, i2, i2, box_hi)

      result(res_hi_ss(X_AXIS),j,k,l,m,n) = box_hi

      RETURN

 999  CALL ef_bail_out (id, err_msg)
      RETURN

      END
