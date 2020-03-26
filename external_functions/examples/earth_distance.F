* 
*  earth_distance.F
* 
*  Ansley manke
*  9/2017
* 
*  Compute the running sum of distance along a path
*  using the law of cosines as in billy's earth_distance.jnl scriopt

      SUBROUTINE earth_distance_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_set_desc(id, 'Return the delta-distance in Km along the path' )

      CALL ef_set_num_args(id, 2)
      CALL EF_SET_HAS_VARI_ARGS(id, NO)
      CALL EF_SET_AXIS_INHERITANCE_6D(id, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL EF_SET_PIECEMEAL_OK_6D(id, NO, NO, NO, NO, NO, NO)

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

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

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


      INTEGER i,j,k,l,m,n
      REAL dlo, dhi, ed_pd, ed_rearth, ed_lat1, ed_lat2, ed_lon1, ed_lon2, 
     .	   ed_x1, ed_x2, ed_y1, ed_y2, ed_z1, ed_z2, ed_d1, ed_angle

      
      ed_pd = 4.*atan(1.)/180.
      ed_rearth = 6371.2 

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* pre-check that the lons array has no missing values
      DO 10 n = arg_lo_ss(F_AXIS,ARG1), arg_hi_ss(F_AXIS,ARG1)
      DO 10 m = arg_lo_ss(E_AXIS,ARG1), arg_hi_ss(E_AXIS,ARG1)
      DO 10 l = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
      DO 10 k = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
      DO 10 j = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
      DO 10 i = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
        IF (lons(i,j,k,l,m,n) .EQ. bad_flag(1)) CALL
     .             EF_BAIL_OUT(id, 'Missing value found in lats')
 10   CONTINUE

* pre-check that the lats array has no missing values
      DO 20 n = arg_lo_ss(F_AXIS,ARG2), arg_hi_ss(F_AXIS,ARG2)
      DO 20 m = arg_lo_ss(E_AXIS,ARG2), arg_hi_ss(E_AXIS,ARG2)
      DO 20 l = arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2)
      DO 20 k = arg_lo_ss(Z_AXIS,ARG2), arg_hi_ss(Z_AXIS,ARG2)
      DO 20 j = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)
      DO 20 i = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)
        IF (lats(i,j,k,l,m,n) .EQ. bad_flag(2)) CALL
     .             EF_BAIL_OUT(id, 'Missing value found in lats')
 20   CONTINUE

* Need to skip the i=1st data

      DO 200 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
      DO 200 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
      DO 200 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
      DO 200 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
      DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
        DO i = res_lo_ss(X_AXIS)+1, res_hi_ss(X_AXIS)  ! faking it, skip

              ed_lat1 = 90 - lats(i-1,j,k,l,m,n)
              ed_lat2 = 90 - lats(i,j,k,l,m,n)
              ed_lon1 = lons(i-1,j,k,l,m,n)
              ed_lon2 = lons(i,j,k,l,m,n)
	      
              ed_x1 = sin( ed_pd*ed_lat1 )* cos( ed_pd* ed_lon1 )
              ed_y1 = sin( ed_pd*ed_lat1 )* sin( ed_pd* ed_lon1 )
              ed_z1 = cos( ed_pd*ed_lat1 )
              ed_x2 = sin( ed_pd*ed_lat2 )* cos( ed_pd* ed_lon2 )
              ed_y2 = sin( ed_pd*ed_lat2 )* sin( ed_pd* ed_lon2 )
              ed_z2 = cos( ed_pd*ed_lat2 )

              ed_d1 = (ed_x1 - ed_x2)**2 + (ed_y1 - ed_y2)**2  +  (ed_z1 - ed_z2)**2
              ed_angle = acos(1 - ed_d1/2)		! angular distance between the points

              result(i,j,k,l,m,n) = ed_rearth* ed_angle
        ENDDO

 200  CONTINUE

      RETURN 
      END

