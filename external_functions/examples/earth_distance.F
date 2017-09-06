* 
*  earth_distance.F
* 
*  Ansley manke
*  9/2017
* 
*  Compute the running sum of distance along a path
*  using the law of cosines as in billy's earth_distance.jnl scriopt
 
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


      SUBROUTINE earth_distance_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 'Return the delta-distance in Km along the path' )

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'longitude')
      CALL ef_set_arg_desc(id, arg,
     .          'List of longitudes in degrees East along the path')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'latitude')
      CALL ef_set_arg_desc(id, arg,
     .          'List of latitudes in degrees North along the path')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

* 
*  In this subroutine we compute the result
* 
      SUBROUTINE earth_distance_compute(id, lons, lats, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL lons(mem1lox:mem1hix, mem1loy:mem1hiy,
     .	   mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL lats(mem2lox:mem2hix, mem2loy:mem2hiy,
     .     mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,j,k,l
      REAL dlo, dhi, ed_pd, ed_rearth, ed_lat1, ed_lat2, ed_lon1, ed_lon2, 
     .	   ed_x1, ed_x2, ed_y1, ed_y2, ed_z1, ed_z2, ed_d1, ed_angle

      
      ed_pd = 4.*atan(1.)/180.
      ed_rearth = 6371.2 

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* pre-check that the lons array has no missing values
      DO 10 l = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
      DO 10 k = arg_lo_ss(Z_AXIS,ARG1), arg_hi_ss(Z_AXIS,ARG1)
      DO 10 j = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
      DO 10 i = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
        IF (lons(i,j,k,l) .EQ. bad_flag(1)) CALL
     .             EF_BAIL_OUT(id, 'Missing value found in lats')
 10   CONTINUE

* pre-check that the lats array has no missing values
      DO 20 l = arg_lo_ss(T_AXIS,ARG2), arg_hi_ss(T_AXIS,ARG2)
      DO 20 k = arg_lo_ss(Z_AXIS,ARG2), arg_hi_ss(Z_AXIS,ARG2)
      DO 20 j = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)
      DO 20 i = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)
        IF (lats(i,j,k,l) .EQ. bad_flag(2)) CALL
     .             EF_BAIL_OUT(id, 'Missing value found in lats')
 20   CONTINUE


* Need to skip the i=1st data


      DO 400 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

        DO 300 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

          DO 200 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            DO 100 i=res_lo_ss(X_AXIS)+1, res_hi_ss(X_AXIS)  ! faking it, skip 
              
              ed_lat1=90-lats(i-1,j,k,l)
              ed_lat2=90-lats(i,j,k,l)
              ed_lon1=lons(i-1,j,k,l)
              ed_lon2=lons(i,j,k,l)

	      
              ed_x1=sin(ed_pd*ed_lat1)*cos(ed_pd*ed_lon1)
              ed_y1=sin(ed_pd*ed_lat1)*sin(ed_pd*ed_lon1)
              ed_z1=cos(ed_pd*ed_lat1)
              ed_x2=sin(ed_pd*ed_lat2)*cos(ed_pd*ed_lon2)
              ed_y2=sin(ed_pd*ed_lat2)*sin(ed_pd*ed_lon2)
              ed_z2=cos(ed_pd*ed_lat2)


              ed_d1=(((ed_x1 - ed_x2)**2) + ((ed_y1 - ed_y2)**2)) + ((ed_z1 - ed_z2)**2)
              ed_angle=acos(1-ed_d1/2)		! angular distance between the points

              result(i,j,k,l) = ed_rearth*ed_angle
 100        CONTINUE

 200      CONTINUE

 300    CONTINUE

 400  CONTINUE
      
         
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

