* 
*  earth_distance.F
* 
*  Ansley manke
*  9/2017
* 
*  Compute the the incremental distance along a path
*  using the law of cosines as in Billy's earth_distance.jnl script

      SUBROUTINE earth_distance_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_set_desc(id, 'Return the point-to-point-distance in Km along the path' )

      CALL ef_set_num_args(id, 2)
      CALL EF_SET_HAS_VARI_ARGS(id, NO)
      CALL EF_SET_AXIS_INHERITANCE_6D(id, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL EF_SET_PIECEMEAL_OK(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'longitude')
      CALL EF_SET_ARG_DESC(id, arg,
     .          'List of longitudes in degrees East along the path')
      CALL EF_SET_AXIS_INFLUENCE_6D(id, arg, YES, YES, YES, YES, YES, YES)

      arg = 2
      CALL EF_SET_ARG_NAME(id, arg, 'latitude')
      CALL EF_SET_ARG_DESC(id, arg,
     .          'List of latitudes in degrees North along the path')
      CALL EF_SET_AXIS_INFLUENCE_6D(id, arg, YES, YES, YES, YES, YES, YES)


      RETURN 
      END

* 
*  Compute the result
* 
      SUBROUTINE earth_distance_compute(id, lons, lats, result)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL lons(mem1lox:mem1hix, mem1loy:mem1hiy,
     .	   mem1loz:mem1hiz, mem1lot:mem1hit,
     .	   mem1loe:mem1hie, mem1lof:mem1hif)
      REAL lats(mem2lox:mem2hix, mem2loy:mem2hiy,
     .     mem2loz:mem2hiz, mem2lot:mem2hit,
     .     mem2loe:mem2hie, mem2lof:mem2hif)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .     memresloz:memreshiz, memreslot:memreshit,
     .     memresloe:memreshie, memreslof:memreshif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(6), res_hi_ss(6), res_incr(6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS), arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr(6,EF_MAX_ARGS)


      INTEGER i, num, nm
      REAL dlo, dhi, ed_pd, ed_rearth, ed_lat1, ed_lat2, ed_lon1, ed_lon2, 
     .	   ed_x1, ed_x2, ed_y1, ed_y2, ed_z1, ed_z2, ed_d1, ed_angle
      CHARACTER*80 errtxt


      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* check that the lists are 1D

      errtxt = 'Argument 1 must be a 1D list'
      num = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1
      DO i = Y_AXIS, F_AXIS
         nm = arg_hi_ss(i,ARG1) - arg_lo_ss(i,ARG1) + 1 
         IF (num.GT.1 .AND. nm.GT.1) GOTO 9000
         num = MAX(num,nm)
      ENDDO

      errtxt = 'Argument 2 must be a 1D list'
      num = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      DO i = Y_AXIS, F_AXIS
         nm = arg_hi_ss(i,ARG2) - arg_lo_ss(i,ARG2) + 1
         IF (num.GT.1 .AND. nm.GT.1) GOTO 9000
         num = MAX(num,nm)
      ENDDO

* Call generalized routine to compute each delta-earth-distance along the path

      errtxt = ' '
      CALL EARTH_DEL (lons, lats, num, bad_flag(1), bad_flag(2), result, errtxt )
      IF (errtxt .NE. ' ') GOTO 9000

      RETURN 
 9000 CALL EF_BAIL_OUT(id, errtxt)

      END

c SUBROUTINE EARTH_DEL is in the source directory fer/utl/
