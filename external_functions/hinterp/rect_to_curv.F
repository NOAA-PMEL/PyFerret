! rect_to_curv.F
! 
! renamed from hinterp_bilinear.F  
! From Matt Harrison hinterp.F
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
! to CALL only the bilinear interpolation modules, and
! recast in F77-compileable code.
! ACM 3/2004
!
! mjh@gfdl.gov
! 2/20/2003
!
! In this subroutine we provide information about
! the function.  The user configurable information 
! consists of the following:
!
! descr              Text description of the function
!
! num_args           Required number of arguments
!
! axis_inheritance   Type of axis for the result
!                       ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
!                       CUSTOM          - user defined axis
!                       IMPLIED_BY_ARGS - same axis as the incoming argument
!                       NORMAL          - the result is normal to this axis
!                       ABSTRACT        - an axis which only has index values
!
! piecemeal_ok       For memory optimization:
!                       axes where calculation may be performed piecemeal
!                       ( YES, NO )
! 
!
! For each argument we provide the following information:
!
! name               Text name for an argument
!
! unit               Text units for an argument
!
! desc               Text description of an argument
!
! axis_influence     Are this argument's axes the same as the result grid?
!                       ( YES, NO )
!
! axis_extend       How much does Ferret need to extend arg limits relative to result 
!


      SUBROUTINE rect_to_curv_init(id)
        
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER   id, arg
      CHARACTER*100  descr

!**********************************************************************
!                                           USER CONFIGURABLE PORTION |
!                                                                     |
!                                                                     V

      WRITE (descr, 10)
   10 FORMAT ('Regrid from rectangular to curvilinear grid')
      CALL ef_set_desc(id, descr)

      CALL ef_set_num_args(id, 4)
      CALL ef_set_num_work_arrays(id, 8)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS,  
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'V')
      CALL ef_set_arg_desc(id, arg,
     ,  'variable V(x,y,z,t) on rectilinear grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'lon_bounds_out')
      CALL ef_set_arg_unit(id, arg, 'degrees')
      CALL ef_set_arg_desc(id, arg, 
     .  'Destination curvilinear grid longitude bounds(2-D)')
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'lat_bounds_out')
      CALL ef_set_arg_unit(id, arg, 'degrees')
      CALL ef_set_arg_desc(id, arg, 
     .  'Destination curvilinear grid latitude bounds(2-D)')
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, NO)   
      
      arg = 4
      CALL ef_set_arg_name(id, arg, 'missing_allowed')
      WRITE (descr, 20)
   20 FORMAT ('number of missing values allowed in four ',
     .   'surrounding source cells: 0 to 3')
      CALL ef_set_arg_desc(id, arg, descr)
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)      

!                                                                     ^
!                                                                     |
!                                           USER CONFIGURABLE PORTION |
!***********************************************************************

      RETURN 
      END  ! subroutine rect_to_curv_init

*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE rect_to_curv_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work array X/Y/Z/T dimensions
*
* ef_set_work_array_lens(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER array_num, nx, ny
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)
      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)

*   real wti(nlon_dst,nlat_dst,2)      
*   weights for bilinear interpolation 

      array_num = 1
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG3), arg_lo_ss(Y_AXIS,ARG3), 
     .     1, 0, 
     .     arg_hi_ss(X_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     2, 0)
     
*  real wtj(nlon_dst,nlat_dst,2)      
*  weights for bilinear interpolation 

      array_num = 2
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG3), arg_lo_ss(Y_AXIS,ARG3), 
     .     1, 0, 
     .     arg_hi_ss(X_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     2, 0)

*  real*8 in_lon_lat(max(nlat_src+1, nlon_src+1), 2) 
*  edges of x,y input axes, will be declared real*8
      array_num = 3
      nx = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 2
      ny = arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1) + 2
      nx = MAX(nx,ny)
      CALL ef_set_work_array_dims(id, array_num,
     .        1,  1, 1, 1,
     .     2*nx, 4, 1, 1)

*  real rad_lon_lat(max(nlat_src+1, nlon_src+1), 2) 
*  edges of input axes, will be declared real, converted to radians

      array_num = 4
      CALL ef_set_work_array_dims(id, array_num,
     .     1,  1, 1, 1,
     .     nx, 2, 1, 1)

*  real out_lon(nlon_dst, nlat_dst)  
*  edges of output longitude grid, will be converted to radians

      array_num = 5
      nx = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      ny = arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1
      CALL ef_set_work_array_dims(id, array_num,
     .     1,   1, 1, 1,
     .     nx, ny, 1, 1)

*  real out_lat(nlon_dst, nlat_dst)      
*  edges of output latitude grid, will be converted to radians

      array_num = 6
      nx = arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3) + 1
      ny = arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3) + 1
      CALL ef_set_work_array_dims(id, array_num,
     .     1,   1, 1, 1,
     .     nx, ny, 1, 1)

*  integer i_lon(nlon_dst,nlat_dst,2) 
* indices for bilinear interpolation 
* These will be declared integer inside the compute routine

      array_num = 7
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG3), arg_lo_ss(Y_AXIS,ARG3), 
     .     1, 1, 
     .     arg_hi_ss(X_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     2, 1)

*  integer j_lat(nlon_dst,nlat_dst,2) 
*  indices for bilinear interpolation 
*  These will be declared integer inside the compute routine

      array_num = 8
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG3), arg_lo_ss(Y_AXIS,ARG3), 
     .     1, 1, 
     .     arg_hi_ss(X_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     2, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


!
! In this subroutine we compute the result
!
      SUBROUTINE rect_to_curv_compute(id, arg_1, arg_2, arg_3, 
     .          arg_4, result, wti, wtj, in_lon_lat,
     .          rad_lon_lat, out_lon, out_lat, i_lon, j_lat)

      implicit none
      
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'
      

      INTEGER  id

      REAL  bad_flag(EF_MAX_ARGS), bad_flag_result

      REAL  arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .       mem1lot:mem1hit)
      REAL  arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz,
     .       mem2lot:mem2hit)
      REAL  arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz,
     .       mem3lot:mem3hit)      
      REAL  arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz,
     .       mem4lot:mem4hit)       

      REAL  result(memreslox:memreshix, memresloy:memreshiy, 
     .      memresloz:memreshiz, memreslot:memreshit)
     
      REAL wti(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL wtj(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL*8 in_lon_lat(wrk3lox:wrk3hix/2, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL rad_lon_lat(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)
      REAL out_lon(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .               wrk5loz:wrk5hiz, wrk5lot:wrk5hit)
      REAL out_lat(wrk6lox:wrk6hix, wrk6loy:wrk6hiy,
     .               wrk6loz:wrk6hiz, wrk6lot:wrk6hit)
      INTEGER i_lon(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .               wrk7loz:wrk7hiz, wrk7lot:wrk7hit)
      INTEGER j_lat(wrk8lox:wrk8hix, wrk8loy:wrk8hiy,
     .               wrk8loz:wrk8hiz, wrk8lot:wrk8hit)

! After initialization, the 'res_' arrays contain indexing information 
! for the result axes.  The 'arg_' arrays will contain the indexing 
! information for each variable's axes. 

      INTEGER  res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER  arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .         arg_incr(4,EF_MAX_ARGS)
      
!**********************************************************************
!                                           USER CONFIGURABLE PORTION |
!                                                                     |
!                                                                     V

      INTEGER  arg, idim, i, j, k, l, interp_method
      INTEGER  i1, j1, k1, l1, k2, k3, l2, l3, verbose,
     .         nlon_src, nlat_src, nlon_dst, nlat_dst, nx, ny

      REAL missing_permit, pi, pi180
      CHARACTER*16 axname(4), ax_units(4) 
      LOGICAL back(4), modulo(4), reg(4)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

! Get edges of coordinate boxes on input axes.  Convert to radians.

      pi =  4.*ATAN(1.)
      pi180 = pi/ 180.

      arg = 1
     
      CALL ef_get_box_limits(id, ARG1, X_AXIS, arg_lo_ss(X_AXIS, ARG1), 
     .   arg_hi_ss(X_AXIS, ARG1), 
     .   in_lon_lat(1,1,1,1), in_lon_lat(1,3,1,1))

! last box upper limit
      nx = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1
      in_lon_lat(nx+1,1,1,1) = in_lon_lat(nx,3,1,1)

c      idim = 1
c      CALL EF_GET_COORDINATES (id, arg, idim,
c     .     arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)+1, 
c     .     in_lon_lat(1,1,1,1) )

! Transform degrees to radians

      nx = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 2
      DO i = 1, nx
        rad_lon_lat(i, 1, 1, 1) = pi180* in_lon_lat(i,1,1,1)
      ENDDO

      CALL ef_get_box_limits(id, ARG1, Y_AXIS, arg_lo_ss(Y_AXIS, ARG1), 
     .   arg_hi_ss(Y_AXIS, ARG1), 
     .   in_lon_lat(1,2,1,1), in_lon_lat(1,4,1,1))

! last box upper limit
      ny = arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1) + 1
      in_lon_lat(ny+1,2,1,1) = in_lon_lat(ny,4,1,1)

c      idim = 2
c      CALL EF_GET_COORDINATES (id, arg, idim,
c     .     arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)+1, 
c     .     in_lon_lat(1,2,1,1) )

! Transform degrees to radians

      ny = arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1) + 2
      DO i = 1, ny
        rad_lon_lat(i, 2, 1, 1) = pi180* in_lon_lat(i,2,1,1)
      ENDDO

      k2 = arg_lo_ss(Z_AXIS,ARG2)
      l2 = arg_lo_ss(T_AXIS,ARG2)
      DO j = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)
         DO i = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)
            out_lon(i, j, 1, 1) = pi180* arg_2(i,j,k2,l2)
         ENDDO
      ENDDO

      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)
      DO j = arg_lo_ss(Y_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3)
         DO i = arg_lo_ss(X_AXIS,ARG3), arg_hi_ss(X_AXIS,ARG3)
            out_lat(i, j, 1, 1) = pi180* arg_3(i,j,k3,l3)
         ENDDO
      ENDDO

      CALL ef_get_axis_info(id,ARG1,axname,ax_units,back,modulo,reg)

      interp_method = 2
!      WRITE(*,*) 'Using bilinear regridding !!'

! Number of missing cells to allow (0 to 3)

      CALL ef_get_one_val (id, 4, missing_permit)
      IF (missing_permit .LT. 0 .OR. missing_permit .GT. 3) 
     .    CALL ef_bail_out (id, 'argument 6 must be between 0 and 3')

      CALL ef_get_bad_flags(id,bad_flag,bad_flag_result)

! nlon_src and nlat_src without the extra element for upper grid box (why?)

      nlon_src = nx-1
      nlat_src = ny-1
      nlon_dst = arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3) + 1
      nlat_dst = arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3) + 1

      verbose = 0
      CALL horiz_interp_init_bili (id, wti, wtj, i_lon, j_lat,  
     .   nlon_src, nlat_src, nlon_dst, nlat_dst, interp_method, 
     .   rad_lon_lat(1,1,1,1), rad_lon_lat(1,2,1,1), out_lon, out_lat, 
     .   verbose, modulo(1) )

      i1 = arg_lo_ss(X_AXIS,ARG1)
      j1 = arg_lo_ss(Y_AXIS,ARG1)
      k1 = arg_lo_ss(Z_AXIS,ARG1)
      l1 = arg_lo_ss(T_AXIS,ARG1)

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)

      DO l = res_lo_ss(T_AXIS),res_hi_ss(T_AXIS)
         DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            CALL horiz_interp_bili ( wti, wtj, i_lon, j_lat,  
     .                    nlon_src, nlat_src, nlon_dst, 
     .                    nlat_dst, interp_method, 
     .                    arg_1(i1,j1,k1,l1), result(i,j,k,l),
     .                    verbose, bad_flag(ARG1), bad_flag_result,
     .                    missing_permit)

            l1 = l1 + arg_incr(T_AXIS,ARG1)
         ENDDO
         k1 = k1 + arg_incr(Z_AXIS,ARG1)
      ENDDO


!                                                                     ^
!                                                                     |
!                                           USER CONFIGURABLE PORTION |
!**********************************************************************

      RETURN 
      END  ! subroutine rect_to_curv_compute
* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*    horiz_interp_init_bili

! From module hinterp_subs_mod


! <CONTACT EMAIL='bw@gfdl.noaa.gov'>
!   Bruce Wyman
! </CONTACT>

! <HISTORY SRC='http://www.gfdl.noaa.gov/fms-cgi-bin/cvsweb.cgi/FMS/'/>

! <OVERVIEW>
!   Performs spatial interpolation between grids.
! </OVERVIEW>

! <DESCRIPTION>
!     This module can interpolate data from rectangular/tripolar grid
!     to rectangular/tripolar grid. Three interpolation schems are used here.
!     when the source grid is tripolar grid, inverse of square distance weighted
!     scheme (spherical regrid) is used. When the source grid is rectangular 
!     longitude/latitude grid, any one of following three schemes can be applied: 
!     conservation scheme that conserves the area-weighed integral of the input field, 
!     bilinear interpolation that use the surround four source grid to interpolate onto
!     the destination grid, spherical regrid that use thes inverse of square distance
!     as weight. User can choose the interpolation method in the horiz_interp_init.
!     when the source grid is rectangular grid (1D), the default method is conserve
!     scheme. When the source grid is tripolar grid (2D), spherical regrid is the 
!     default algorithm and no other scheme can be applied. 

!     There is an optional mask field for missing input data.
!     An optional output mask field may be used in conjunction with
!     the input mask to show where output data exists.
! </DESCRIPTION>

!-----------------------------------------------------------------------
!
!        Performs spatial interpolation between grids.
!
!-----------------------------------------------------------------------

      


!---- interfaces ----



! <INTERFACE NAME='horiz_interp_init'>
!   <OVERVIEW>
!      Allocates space and initializes a derived-type variable
!      that contains pre-computed interpolation indices and weights.
!   </OVERVIEW>
!   <DESCRIPTION>
!      Allocates space and initializes a derived-type variable
!      that contains pre-computed interpolation indices and weights
!      for improved performance of multiple interpolations between
!      the same grids. This routine does not need to be called if you
!      are doing a single grid-to-grid interpolation.
!   </DESCRIPTION>
!   <IN NAME='lon_in' TYPE='real' DIM='dimension(:), dimension(:,:)' UNITS='radians'>
!      Longitude (in radians) for source data grid. when lon_in is 1D, it is the longitude
!      edges and its value are for adjacent grid boxes and must increase 
!      in value. When lon_in is 2D, there are two cases: one is the longitude edges stored as
!      pairs for each grid box (when interp_method is 1(conservative)), the other is the longitude
!      of the center of each grid box. 
!   </IN>
!   <IN NAME='lat_in' TYPE='real' DIM='dimension(:), dimension(:,:)' UNITS='radians'>
!      Latitude (in radians) for source data grid. when lat_in is 1D, it is the latitude
!      edges and its value are for adjacent grid boxes and must increase 
!      in value. When lat_in is 2D, it is the longitude of the center of each grid box.
!      When interp_method is 1(conservative) or 2(bilinear), lon_in should be 1D.
!   </IN>
!   <IN NAME='lon_out' TYPE='real' DIM='dimension(:), dimension(:,:)' UNITS='radians' >
!      Longitude (in radians) for source data grid. when lon_in is 1D, it is the longitude
!      edges and its value are for adjacent grid boxes and must increase 
!      in value. When lon_in is 2D, there are two cases: one is the longitude edges stored as
!      pairs for each grid box (when interp_method is 1), the other is the longitude
!      of the center of each grid box (when interp_method is 2). 
!   </IN>
!   <IN NAME='lat_out' TYPE='real' DIM='dimension(:), dimension(:,:)' UNITS='radians' >
!      Latitude (in radians) for source data grid. when lat_in is 1D, it is the latitude
!      edges and its value are for adjacent grid boxes and must increase 
!      in value. When lat_in is 2D, there are two cases: one is the latitude edges stored as
!      pairs for each grid box (when interp_method is 1), the other is the latitude
!      of the center of each grid box (when interp_method is 2).
!   </IN>
!   <IN NAME='verbose' TYPE='integer'>
!      Integer flag that controls the amount of printed output.
!      verbose = 0, no output; = 1, min,max,means; = 2, still more
!   </IN>
!   <IN NAME='interp_method' TYPE='integer'> 
!      interpolation method, = 1, using conservation scheme,
!      = 2, using bilinear interpolation, = SPHERICA(3),using spherical regrid.
!      when source grid is 1d, default value is CONSERVE; when source grid is 2d,
!      default value is 'SPHERICA'.
!   </IN>
!   <IN NAME = 'src_modulo' >
!      Indicate the source data grid is cyclic or not.
!   </IN>
!   <OUT NAME='Interp' >
!      A derived-type variable containing indices and weights used for subsequent 
!      interpolations. To reinitialize this variable for a different grid-to-grid 
!      interpolation you must first use the 'horiz_interp_end' interface.
!   </OUT>


! </INTERFACE>


C      real wti(nlon_dst,nlon_dst,2)      !weights for bilinear interpolation 
C      real wtj(nlat_dst,nlat_dst,2)
C      integer i_lon(nlon_dst,nlon_dst,2) !indices for bilinear interpolation 
C      integer j_lat(nlat_dst,nlat_dst,2)
C      integer nlon_src, nlat_src         !size of source grid
C      integer nlon_dst, nlat_dst         !size of destination grid
C      integer interp_method              !interpolation method.
                                         !  =1, conservative scheme
                                         !  =2, bilinear interpolation
                                         !  =3, spherical regrid


C Interp list:
C wti, wtj, i_lon, j_lat, nlon_src, nlat_src, nlon_dst, nlat_dst, interp_method 

C Interp allocation from subroutine horiz_interp_init_bili 

      subroutine horiz_interp_init_bili (id, wti, wtj, i_lon, j_lat,  
     .   nlon_src, nlat_src, nlon_dst,nlat_dst, interp_method, 
     .   lon_in, lat_in, lon_out, lat_out, verbose, src_modulo )

!-----------------------------------------------------------------------
C inputs
C  From type horiz_interp_type, variable Interp


      real wti(nlon_dst,nlat_dst,2)      !weights for bilinear interpolation 
      real wtj(nlon_dst,nlat_dst,2)
      integer i_lon(nlon_dst,nlat_dst,2) !indices for bilinear interpolation 
      integer j_lat(nlon_dst,nlat_dst,2)
      integer nlon_src, nlat_src         !size of source grid
      integer nlon_dst, nlat_dst         !size of destination grid
      integer interp_method              !interpolation method.

C Other inputs

      real lon_in(*) , lat_in(*)
      real lon_out(nlon_dst,nlat_dst), lat_out(nlon_dst,nlat_dst)
      integer id, verbose
      logical src_modulo
!-----------------------------------------------------------------------
C  Other declarations

      integer INDP, nlon_in, nlat_in, nlon_out, nlat_out, n, m,
     .   i, j, ie, is, je, js, istart, jstart, iend, jend, 
     .   ln_err, lt_err, na, warns
      real wtw, wte, wts, wtn, lon, lat, pi, tpi, hpi,      
     .   glt_min, glt_max, gln_min, gln_max, min_lon, max_lon

! parameters
      real epsln, large
      integer stdout 
      parameter (epsln=1.e-10, large=1.e20, stdout = 6)

      pi  =  3.14159265358979323846 
      hpi = 0.5*pi
      tpi = 4.0*hpi
      glt_min = hpi
      glt_max = -hpi
      gln_min = tpi
      gln_max = -tpi
      min_lon = 0.0
      max_lon = tpi
      ln_err = 0
      lt_err = 0

      warns = verbose

      nlon_in = nlon_src  
      nlat_in = nlat_src
      nlon_out = nlon_dst
      nlat_out = nlat_dst

      if(lon_in(nlon_in) - lon_in(1) .gt. tpi + epsln) 
     . call ef_bail_out (id, 
     . 'The range of source grid longitude may be no larger than tpi')

      if(lon_in(1) .lt. 0.0) then
        min_lon = lon_in(1)
        max_lon = lon_in(nlon_in)
      endif

!     find longitude points of data within interval [-360., 360.]
      istart = 1
      do i=2,nlon_in
        if (lon_in(i-1) .lt. -tpi .and. lon_in(i) .ge. -tpi) istart = i
      enddo 
      iend = nlon_in
      do i=2,nlon_in
        if (lon_in(i-1) .lt. tpi  .and. lon_in(i) .ge. tpi) iend = i
      enddo

!     find latitude points of data within interval [-90., 90.]
      jstart = 1
      do j=2,nlat_in
        if (lat_in(j-1) .lt. -1.0 * hpi .and. 
     .      lat_in(j  ) .ge. -1.0 * hpi) jstart = j
      enddo 
      jend = nlat_in
      do j=2,nlat_in
        if (lat_in(j-1) .lt. hpi .and. lat_in(j) .ge.hpi ) jend = j
      enddo

      do n = 1, nlat_out
      do m = 1, nlon_out
        lon = lon_out(m,n)
        lat = lat_out(m,n)

        if(lon .lt. min_lon) then
         lon = lon + tpi
        else if(lon .gt. max_lon) then
         lon = lon - tpi
        endif
! when the input grid is in not cyclic, the output grid should located inside
! the input grid
      if(.not. src_modulo) then
        if((lon .lt. lon_in(1)) .or. (lon .gt. lon_in(nlon_in))) 
     .   call ef_bail_out(id, 
     .'input grid not modulo, output grid must be inside input grid')


      endif

      glt_min = min(lat,glt_min)
      glt_max = max(lat,glt_max)
      gln_min = min(lon,gln_min)
      gln_max = max(lon,gln_max)

      na = iend - istart + 1
      is = INDP(lon, lon_in(istart), na ) + istart - 1
      if( lon_in(is) .gt. lon ) is = max(is - 1,istart)
      if( lon_in(is) .eq. lon .and. is .eq. nlon_in) 
     .      is = max(is - 1,istart)
      ie = min(is+1,iend)
      if(lon_in(is) .ne. lon_in(ie) .and. lon_in(is) .le. lon) then
        wtw = ( lon_in(ie) - lon) / (lon_in(ie) - lon_in(is) )
      else
! east or west of the last data value. this could be because a
! cyclic condition is needed or the dataset is too small. 
        ln_err = 1
        ie = istart
        is = iend
        if (lon_in(ie) .ge. lon ) then
           wtw = (lon_in(ie) -lon)/(lon_in(ie)-lon_in(is)+tpi+epsln)
        else
          wtw = (lon_in(ie) -lon+tpi+epsln)/
     .          (lon_in(ie)-lon_in(is)+tpi+epsln)
        endif   
      endif
      wte = 1. - wtw

      na = jend - jstart + 1
      js = INDP(lat, lat_in(jstart), na ) + jstart - 1

      if( lat_in(js) .gt. lat ) js = max(js - 1, jstart)
      if( lat_in(js) .eq. lat .and. js .eq. jend) 
     .     js = max(js - 1, jstart)
      je = min(js + 1, jend)

      if ( lat_in(js) .ne. lat_in(je) .and. lat_in(js) .le. lat) then
        wts = ( lat_in(je) - lat )/(lat_in(je)-lat_in(js))
      else
!     north or south of the last data value. this could be because a
!     pole is not included in the data set or the dataset is too small.
!     in either case extrapolate north or south
         lt_err = 1
         wts = 1.
      endif

      wtn = 1. - wts
      
            i_lon (m,n,1) = is
            i_lon (m,n,2) = ie
            j_lat (m,n,1) = js
            j_lat (m,n,2) = je
            wti   (m,n,1) = wtw
            wti   (m,n,2) = wte
            wtj   (m,n,1) = wts
            wtj   (m,n,2) = wtn
      
      enddo
      enddo
      if (ln_err .eq. 1 .and. warns .GT. 0) then
        write (stdout,'(/,(1x,a))') 
     . '==> Warning: the geographic data set does not extend far   ', 
     . '             enough east or west - a cyclic boundary       ', 
     . '             condition was applied. check if appropriate   '
         write (stdout,'(/,(1x,a,2f8.4))') 
     . '    data required between longitudes:', gln_min, gln_max,     
     . '    data set is between longitudes:', 
     .   lon_in(istart), lon_in(iend)
         warns = warns - 1
       endif
!
       if (lt_err .eq. 1 .and. warns .GT. 0) then
         write (stdout,'(/,(1x,a))')
     . '==> Warning: the geographic data set does not extend far   ',
     . '             enough north or south - extrapolation from    ',
     . '             the nearest data was applied. this may create ',
     . '             artificial gradients near a geographic pole   ' 
         write (stdout,'(/,(1x,a,2f8.4))')
     . '    data required between latitudes:', glt_min, glt_max,
     . '      data set is between latitudes:', 
     .   lat_in(jstart), lat_in(jend)
         warns = warns - 1
       endif

       return

      end ! subroutine horiz_interp_init_bili

* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine horiz_interp_bili ( wti, wtj, i_lon, j_lat,  
     .                    nlon_src, nlat_src, nlon_dst, 
     .                    nlat_dst, interp_method, 

     .                    data_in, data_out, verbose, 
     .                    missing_value,  missing_result, 
     .                    missing_permit)
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
C  From type horiz_interp_type, variable Interp

      real wti(nlon_dst,nlat_dst,2)      !weights for bilinear interpolation 
      real wtj(nlon_dst,nlat_dst,2)
      integer i_lon(nlon_dst,nlat_dst,2) !indices for bilinear interpolation 
      integer j_lat(nlon_dst,nlat_dst,2)
      integer nlon_src, nlat_src         !size of source grid
      integer nlon_dst, nlat_dst         !size of destination grid
      integer interp_method              !interpolation method.

      real data_in(nlon_src, nlat_src)   ! input
      real data_out(nlon_dst, nlat_dst)  ! output
      integer verbose
      real missing_value                 ! missing flag of input
      real missing_result                ! missing flag for result
      real missing_permit                ! Number of missing cells to allow
!-----------------------------------------------------------------------
      integer nlon_in, nlat_in, nlon_out, nlat_out, n, m,
     .            is, ie, js, je, iverbose, num_missing, 
     .            miss_in, miss_out
      real  dwtsum, wtsum, min_in, max_in, avg_in, 
     .            min_out, max_out, avg_out, wtw, wte, wts, wtn
      real data(4)

! parameters
      real epsln, large
      parameter (epsln=1.e-10, large=1.e20)

      nlon_in  = nlon_src
      nlat_in  = nlat_src
      nlon_out = nlon_dst
      nlat_out = nlat_dst

      iverbose = verbose

      do n = 1, nlat_out
      do m = 1, nlon_out
      data_out(m,n) = 0.0

      is = i_lon (m,n,1) 
      ie = i_lon (m,n,2)
      js = j_lat (m,n,1)
      je = j_lat (m,n,2)
      wtw = wti(m,n,1)
      wte = wti(m,n,2)
      wts = wtj(m,n,1)
      wtn = wtj(m,n,2)
      data(1) = data_in(is,js)
      data(2) = data_in(ie,js)
      data(3) = data_in(ie,je)
      data(4) = data_in(is,je)

      call data_sum_bilinear ( data, wtw,wte, wts,wtn, dwtsum, wtsum, 
     .   num_missing, missing_value)

      if (num_missing .gt. missing_permit .or. wtsum .lt. epsln) then
        data_out(m,n) = missing_result
      else
        data_out(m,n) = dwtsum/wtsum
      endif
      enddo
      enddo

!***********************************************************************
! compute statistics: minimum, maximum, and mean
!-----------------------------------------------------------------------

!!      if (iverbose .GT. 0) then
!!
!!! compute statistics of input data
!!
!!       call stats_type2 (data_in, min_in, max_in, avg_in, 
!!   .           miss_in, missing_value)
!!
!!! compute statistics of output data
!!       call stats_type2 (data_out, min_out, max_out, avg_out, 
!!   .          miss_out, missing_value)
!!
!!!---- output statistics ----
!!! root_pe have the information of global mean, min and max
!!
!!       write (*,900)
!!       write (*,901)  min_in ,max_in, avg_in
!!         write (*,903)  miss_in
!!         write (*,902)  min_out,max_out,avg_out
!!         write (*,903)  miss_out
!! 900     format (/,1x,10('-'),' output from horiz_interp ',10('-'))
!! 901     format ('  input:  min=',f16.9,'  max=',f16.9,'  avg=',f22.15)
!! 902     format (' output:  min=',f16.9,'  max=',f16.9,'  avg=',f22.15)
!! 903     format ('          number of missing points = ',i6)
!!
!!      endif


      return

      end ! subroutine horiz_interp_bili

* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

       integer function INDP (value, array, ia)
       real array(*)
       real value
       integer ia
!
!=======================================================================
!
!     INDP = index of nearest data point within "array" corresponding to
!            "value".

!     inputs:
!     value  = arbitrary data...same units as elements in "array"
!     array  = array of data points  (must be monotonically increasing)

!     output:
!     INDP =  index of nearest data point to "value"
!             if "value" is outside the domain of "array" then INDP = 1
!             or "ia" depending on whether array(1) or array(ia) is
!             closest to "value"
!=======================================================================
!
       integer i
       logical keep_going
!
       do i=2,ia
         if (array(i) .lt. array(i-1)) then
         write (6,*) 
     .   ' => Error: array must be monotonically increasing in "INDP"', 
     .   '          when searching for nearest element to value=',value
         write (6,*) '           array(i) < array(i-1) for i=',i 
         write (6,*) '           array(i) for i=1..ia follows:'
         stop
        endif
      enddo
      if (value .lt. array(1) .or. value .gt. array(ia)) then
        if (value .lt. array(1))  INDP = 1
        if (value .gt. array(ia)) INDP = ia
      else
        i=1
        keep_going = .true.
        do while (i .le. ia .and. keep_going)
          i = i+1
          if (value .le. array(i)) then
            INDP = i
            if (array(i)-value .gt. value-array(i-1)) INDP = i-1
            keep_going = .false.
          endif
        enddo
      endif
      return
      end ! function INDP

* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine data_sum_bilinear ( data, wtw, wte, wts, wtn, 
     .                               dwtsum, wtsum, num_missing,
     .                               missing_value  )

!  sums up the data and weights for a single output grid box
!-----------------------------------------------------------------------
      real data(*)
      real wtw, wte, wts, wtn
      real dwtsum, wtsum
      integer num_missing
      real missing_value

!  dwtsum = sum(data*wt*mask)
!  wtsum  = sum(wt*mask)

      integer i
      real wt1, wt2
!-----------------------------------------------------------------------


      num_missing = 0
      dwtsum = 0.0
      wtsum  = 0.0

      do i = 1,4
      if(data(i) .eq. missing_value) then
         num_missing = num_missing + 1
      else
         if (i .eq. 1) then
            wt1= wtw
            wt2 = wts
         else if (i .eq. 2) then
            wt1= wte
            wt2 = wts
         else if (i .eq. 3) then
            wt1= wte
            wt2 = wtn
         else if (i  .eq. 4) then
            wt1= wtw 
            wt2 = wtn
         endif

         dwtsum = dwtsum + data(i) * wt1 * wt2
         wtsum  = wtsum + wt1 * wt2
      endif
      enddo

      return
      end ! subroutine data_sum_bilinear

* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

!---This statistics is for bilinear interpolation and spherical regrid.

!!      subroutine stats_type2 ( dat, low, high, avg, miss, missing_value,
!!     .  nlon, nlat)
!!
!!      real dat(nlon,nlat)
!!      real low, high, avg  ! output
!!      integer miss         ! output
!!      real  missing_value  ! input
!!      integer nlon, nlat   ! input
!!
!!
!!      real dsum, npts
!!      integer n, m
!!
!!      dsum = 0.0
!!      miss = 0
!!      npts = 0
!!
!!      do n = 1, nlat
!!      do m = 1, nlon
!!      IF (dat(m,n) .NE. missing_value) THEN
!!         npts = npts + 1
!!         low = MIN( low, dat(m,n) )
!!         high = MAX (high, dat(m,n) )
!!         dsum = dsum + dat(m,n)
!!      ELSE
!!         miss = miss + 1
!!      ENDIF
!!      ENDDO
!!      ENDDO
!!
!!      if(npts .EQ. 0) then
!!         avg = missing_value
!!      else
!!         avg = dsum/real(npts)
!!      endif
!!   
!!      return
!!
!!      end  ! subroutine stats_type2


* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
