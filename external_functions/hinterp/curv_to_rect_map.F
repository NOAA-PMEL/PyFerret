! curv_to_rect_map.F  
! From Matt Harrison hinterp.F
! to call only the spherical interpolation modules, and
! recast in F77-compileable format.
! ACM 3/2004
!  10/8/2004  add fourth argument radius  (max_dist) so this can be
!             set to a smaller value for fine grids or small regions.
!
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
! In this SUBROUTINE we provide information about
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


      SUBROUTINE curv_to_rect_map_init(id)
        
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER   id, arg
      CHARACTER*100  descr

!**********************************************************************
!                                           USER CONFIGURABLE PORTION |
!                                                                     |
!                                                                     V

      WRITE (descr, 10)
   10 FORMAT
     . ('Compute mapping for regridding: curvilinear to ',
     .  'rectangular grid.')
      CALL ef_set_desc(id, descr)

      CALL ef_set_num_args(id, 4)
      CALL ef_set_num_work_arrays(id, 9)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS,  
     .     IMPLIED_BY_ARGS, ABSTRACT, ABSTRACT)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'lon_in')
      CALL ef_set_arg_unit(id, arg, 'degrees')
      CALL ef_set_arg_desc(id, arg, 
     .   'Source grid longitudes (2-D)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'lat_in')
      CALL ef_set_arg_unit(id, arg, 'degrees')
      CALL ef_set_arg_desc(id, arg, 
     .  'Source grid latitudes (2-D)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'grid_out')
      CALL ef_set_arg_unit(id, arg, 'degrees')
      WRITE (descr, 20)
   20 FORMAT ('Any variable on destination Longitude-Latitude ',
     .        'grid, Lon and Lat')
      CALL ef_set_arg_desc(id, arg, descr)
     .  
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'radius')
      CALL ef_set_arg_unit(id, arg, 'degrees')
      WRITE (descr, 30)
   30 FORMAT ('Source points falling ',
     .   'within radius are included in mapping to destination point')
      CALL ef_set_arg_desc(id, arg, descr)
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

!                                                                     ^
!                                                                     |
!                                           USER CONFIGURABLE PORTION |
!***********************************************************************

      RETURN 
      END  ! SUBROUTINE curv_to_rect_map_init

* Set the limits of the abstract axes for the result.

      SUBROUTINE curv_to_rect_map_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      call ef_set_axis_limits(id, Z_AXIS, 1, 4)
      call ef_set_axis_limits(id, T_AXIS, 1, 4)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END  ! SUBROUTINE curv_to_rect_map_result_limits



*
* In this SUBROUTINE we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE curv_to_rect_map_work_size(id)

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
      INTEGER array_num
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)
      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)

      INTEGER num_nbrs, nx, ny

      num_nbrs = 4

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)


* Define together as one work array:
* i_lon_j_lat(nlon_dst, nlat_dst, num_nbrs, 2)
*  integer i_lon(nlon_dst, nlat_dst, num_nbrs) 
*  integer j_lat(nlon_dst, nlat_dst, num_nbrs) 
*  indices for spherical interpolation - define together as one work array

      array_num = 1
      num_nbrs = 4
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG3), arg_lo_ss(Y_AXIS,ARG3), 
     .     1, 1, 
     .     arg_hi_ss(X_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     num_nbrs, 2)


*  Define together as one work array:
*  real src_dist(nlon_dst, nlat_dst, num_nbrs)
*  real map_src_dist(nlon_dst, nlat_dst, num_nbrs)

      array_num = 2
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG3), arg_lo_ss(Y_AXIS,ARG3), 
     .     1, 1, 
     .     arg_hi_ss(X_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     num_nbrs, 2)

* Define together as one work array:
* logical found_neighbors(nlon_dst, nlat_dst)
* logical map_found_neighbors(nlon_dst, nlat_dst)

      array_num = 3
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG3), arg_lo_ss(Y_AXIS,ARG3), 
     .     1, 1, 
     .     arg_hi_ss(X_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     2, 1)

* integer map_src_add(nlon_dst, nlat_dst, num_nbrs)

      array_num = 4
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG3), arg_lo_ss(Y_AXIS,ARG3), 
     .     1, 1, 
     .     arg_hi_ss(X_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     num_nbrs, 1)

* real wt(nlon_dst, nlat_dst, num_nbrs) 

      array_num = 5
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG3), arg_lo_ss(Y_AXIS,ARG3), 
     .     1, 1, 
     .     arg_hi_ss(X_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     num_nbrs, 1)

* real  mask_dst(nlon_dst, nlat_dst) 

      array_num = 6
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG3), arg_lo_ss(Y_AXIS,ARG3), 
     .     1, 1, 
     .     arg_hi_ss(X_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     1, 1)

* Define together as one work array:
* real src_grid_radians(nx_src, ny_src, 2, 1)
* source grid longitudes (arg_1), translated to radians
* source grid latitudes  (arg_2), translated to radians

      array_num = 7
      CALL ef_set_work_array_dims(id, array_num,
     .     arg_lo_ss(X_AXIS,ARG1), arg_lo_ss(Y_AXIS,ARG2), 1, 1, 
     .     arg_hi_ss(X_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG2), 2, 1)

* real*8 dst_grid(MAX(nx_dst, ny_dst), 2, 1, 1)
* destination rectangular grid, translated to radians
* allocate 2* needed number of values, since this will be real*8

      array_num = 8
      nx =  arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3) + 1
      ny =  arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3) + 1
      nx = MAX(nx,ny)
      CALL ef_set_work_array_dims(id, array_num,
     .        1,   1, 1, 1, 
     .     2*nx, 2*2, 1, 1)

* real dst_grid_radians(nx_dst, ny_dst, 2, 1)
* destination rectangular grid, translated to radians

      array_num = 9
      nx =  arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3) + 1
      ny =  arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3) + 1
      CALL ef_set_work_array_dims(id, array_num,
     .      1,  1, 1, 1, 
     .     nx, ny, 2, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

!
! In this SUBROUTINE we compute the result
!
      SUBROUTINE curv_to_rect_map_compute(id, arg_1, arg_2, arg_3,
     .          arg_4, result, i_lon_j_lat, src_dist, 
     .          found_neighbors, map_src_add, wt, mask_dst, 
     .          src_grid_radians, dst_grid, dst_grid_radians)

! Schematic:
!    The following example illustrates a destination grid location (+) with 
!    a (R)adius of influence (in radians) denoted by (=).  Valid source grid 
!    locations (o) which fall within the radius of influence of the destination 
!    point are used in the mapping.  Masked points (x) do not contribute to the mapping
!    but are included as nearest neighbors. In this case, 4 valid source grid points fall
!    within the radius of influence.   
!
! 
!               o  o   o   o   o   o   o
!                      =========
!                     =         = 
!               o  o = o   o   o = o   o
!                   =        R    = 
!                   =      +----->=
!               o  o   o   x   x   x   x
!                     =         =
!                      =========
!               o  o   o   x   x   x   x
!
!
!
      IMPLICIT none
      
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
     
* Declare work arrays

      INTEGER i_lon_j_lat(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .              wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL src_dist(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .              wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      LOGICAL found_neighbors(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .              wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      INTEGER map_src_add(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .              wrk4loz:wrk4hiz, wrk4lot:wrk4hit)
      REAL wt(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .              wrk5loz:wrk5hiz, wrk5lot:wrk5hit)
      REAL mask_dst(wrk6lox:wrk6hix, wrk6loy:wrk6hiy,
     .              wrk6loz:wrk6hiz, wrk6lot:wrk6hit)
      REAL src_grid_radians(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .              wrk7loz:wrk7hiz, wrk7lot:wrk7hit)
      REAL*8 dst_grid(wrk8lox:wrk8hix/2, wrk8loy:wrk8hiy/2,
     .              wrk8loz:wrk8hiz, wrk8lot:wrk8hit)
      REAL dst_grid_radians(wrk9lox:wrk9hix, wrk9loy:wrk9hiy,
     .              wrk9loz:wrk9hiz, wrk9lot:wrk9hit)

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

      INTEGER  i, j, k, l, arg, idim, interp_method, nx, ny,
     .         nlon_src, nlat_src, nlon_dst, nlat_dst, num_nbrs, 
     .         ilon(4), jlat(4)

! ilon and jlat are size of num_nbrs

      REAL max_dist, pi, pi180
      LOGICAL src_modulo

      pi =  4.*ATAN(1.)
      pi180 = pi/ 180.

      num_nbrs = 4

!  Get max_dist as an argument and convert to radians.
cc      max_dist = 0.17
      arg = 4
      CALL ef_get_one_val(id, arg, max_dist)
      max_dist = pi180*max_dist

      src_modulo = .true.  ! Do we want this? Ferret does not now deal w/ modulo curvi data

!  Loop over x,y,t of the input fields
      
      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      interp_method = 3
!      WRITE(*,*) 'Using spherical regridding !!'

      CALL ef_get_bad_flags(id,bad_flag,bad_flag_result)

      nlon_src = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1
      nlat_src = arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1) + 1
      nlon_dst = arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3) + 1
      nlat_dst = arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3) + 1

      nx = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      ny = arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1

      IF (nlon_src .NE. nx .OR. nlat_src .NE. ny) call ef_bail_out (id, 
     . 'argument 2 must have the same x and y dimensions as argument 1')  

* Tranlate input and output grid coordinates to radians.

* arg_1 is lons_in
* arg_2 is lats_in
* src_grid_radians(nx_src, ny_src, 2, 1)

      pi =  4.*ATAN(1.)
      pi180 = pi/ 180.

      DO j = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
      DO i = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
         src_grid_radians(i,j,1,1) = pi180*arg_1(i,j,mem1loz,mem1lot)
      ENDDO
      ENDDO

      DO j = arg_lo_ss(Y_AXIS,ARG2), arg_hi_ss(Y_AXIS,ARG2)
      DO i = arg_lo_ss(X_AXIS,ARG2), arg_hi_ss(X_AXIS,ARG2)
         src_grid_radians(i,j,2,1) = pi180*arg_2(i,j,mem2loz,mem2lot)
      ENDDO
      ENDDO

      arg = 3
      idim = 1

      CALL EF_GET_COORDINATES (id, arg, idim,
     .     arg_lo_ss(X_AXIS,ARG3), arg_hi_ss(X_AXIS,ARG3), 
     .     dst_grid(1,1,1,1) )

      arg = 3
      idim = 2

      CALL EF_GET_COORDINATES (id, arg, idim,
     .     arg_lo_ss(Y_AXIS,ARG3), arg_hi_ss(Y_AXIS,ARG3), 
     .     dst_grid(1,2,1,1) )

      DO j = 1, nlat_dst
      DO i = 1, nlon_dst
        dst_grid_radians(i, j, 1, 1) = pi180* dst_grid(i,1,1,1)
      ENDDO
      ENDDO

      DO j = 1, nlat_dst
      DO i = 1, nlon_dst
        dst_grid_radians(i, j, 2, 1) = pi180* dst_grid(j,2,1,1)
      ENDDO
      ENDDO

      CALL horiz_interp_init_sphe(id, 
     .        src_grid_radians(wrk7lox,wrk7loy,1,1), 
     .        src_grid_radians(wrk7lox,wrk7loy,2,1), 
     .        dst_grid_radians(1,1,1,1), dst_grid_radians(1,1,2,1),
     .        i_lon_j_lat(1,1,1,1), i_lon_j_lat(1,1,1,2), 
     .        ilon, jlat, 
     .        src_dist(wrk2lox,wrk2loy,wrk2loz,1), 
     .        src_dist(wrk2lox,wrk2loy,wrk2loz,2),
     .        found_neighbors(1,1,1,1), map_src_add, 
     .        found_neighbors(1,1,2,1), num_nbrs, max_dist, 
     .        src_modulo, nlon_src, nlat_src, nlon_dst, nlat_dst)

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)

      CALL save_mapping (src_dist, wt, mask_dst, 
     .        found_neighbors(1,1,1,1), i_lon_j_lat(1,1,1,1),  
     .        i_lon_j_lat(1,1,1,2), max_dist, nlon_src, nlat_src, 
     .        nlon_dst, nlat_dst, num_nbrs, bad_flag_result,
     .        result(i,j,k,l) )

!                                                                     ^
!                                                                     |
!                                           USER CONFIGURABLE PORTION |
!**********************************************************************

      RETURN 
      END  ! SUBROUTINE curv_to_rect_map_compute

      
      subroutine horiz_interp_init_sphe(id, lon_in, lat_in, lon_out, 
     .        lat_out, i_lon, j_lat, ilon, jlat, src_dist, map_src_dist,
     .        found_neighbors, map_src_add, map_found_neighbors, 
     .        num_nbrs, max_dist, src_modulo, nlon_src, nlat_src, 
     .        nlon_dst, nlat_dst)
      
      integer id, nlon_src, nlat_src, nlon_dst, nlat_dst,  
     .        num_nbrs, ilon(num_nbrs), jlat(num_nbrs), 
     .        map_src_add(nlon_dst,nlat_dst,num_nbrs),
     .        i_lon(nlon_dst,nlat_dst,num_nbrs),
     .        j_lat(nlon_dst,nlat_dst,num_nbrs)

      real    max_dist, 
     .        lon_in(nlon_src, nlat_src),  lat_in(nlon_src, nlat_src), 
     .        lon_out(nlon_dst, nlat_dst), lat_out(nlon_dst, nlat_dst),
     .        src_dist(nlon_dst,nlat_dst,num_nbrs)

      logical found_neighbors(nlon_dst,nlat_dst), src_modulo
      
!------local variables ---------------------------------------
      
      integer  i, j, n, num_neighbors, map_dst_xsize, map_dst_ysize, 
     .         map_src_xsize, map_src_ysize, map_src_size
      real     sum, max_src_dist,
     .         map_src_dist(nlon_dst,nlat_dst,num_nbrs)
      
      logical map_found_neighbors(nlon_dst, nlat_dst)

!--------------------------------------------------------------
      map_dst_xsize = nlon_dst
      map_dst_ysize = nlat_dst
      map_src_xsize = nlon_src
      map_src_ysize = nlat_src
      map_src_size = map_src_xsize*map_src_ysize
      
      num_neighbors = num_nbrs
      max_src_dist = max_dist
      
!using radial_search to find the nearest points and corresponding distance.

      call radial_search(id, lon_in, lat_in, lon_out, lat_out, 
     .          map_src_add, map_src_dist, map_found_neighbors,
     .          num_nbrs, max_dist, src_modulo, nlon_src, nlat_src,
     .          nlon_dst, nlat_dst, map_src_size )
      
      do j=1,map_dst_ysize
      do i=1,map_dst_xsize
         sum=0.0
         do n=1,num_neighbors
            if(map_src_add(i,j,n) .EQ. 0) then
               jlat(n) = 0
               ilon(n) = 0
            else
               jlat(n) = map_src_add(i,j,n)/map_src_xsize + 1
               ilon(n) = map_src_add(i,j,n) - (jlat(n)-1)*map_src_xsize
               if(ilon(n) .EQ. 0) then
                  jlat(n) = jlat(n) - 1
                  ilon(n) = map_src_xsize
               endif
            endif
            i_lon(i,j,n) = ilon(n)
            j_lat(i,j,n) = jlat(n)
         enddo
      
      enddo
      enddo
      
      do j=1,map_dst_ysize
      do i=1,map_dst_xsize
         do n=1,num_neighbors
            src_dist(i,j,n) = map_src_dist(i,j,n)
         enddo
         found_neighbors(i,j) = map_found_neighbors(i,j)
      enddo
      enddo
      
      nlon_src = map_src_xsize
      nlat_src = map_src_ysize
      nlon_dst = map_dst_xsize
      nlat_dst = map_dst_ysize
      
      return

      end ! subroutine horiz_interp_init_sphe
!#####################################################################
      
       subroutine radial_search(id, theta_src, phi_src, theta_dst, 
     .      phi_dst, map_src_add, map_src_dist, map_found_neighbors, 
     .      num_nbrs, max_dist, src_modulo, nlon_src, nlat_src,
     .      nlon_dst, nlat_dst, map_src_size)
      
! x_dst,y_dst = destination grid lon,lat
! x_src,y_src = source grid lon,lat


!       theta_src = reshape(x_src,(/map_src_size/))
!       phi_src = reshape(y_src,(/map_src_size/))
!       theta_dst(:,:) = x_dst(:,:)
!       phi_dst(:,:) = y_dst(:,:)      

       integer id, nlon_src, nlat_src, nlon_dst, nlat_dst, num_nbrs, 
     .         map_src_add(nlon_dst,nlat_dst,num_nbrs)

       real    theta_src(map_src_size), phi_src(map_src_size),  
     .         theta_dst(nlon_dst, nlat_dst), 
     .         phi_dst(nlon_dst, nlat_dst),
     .         map_src_dist(nlon_dst,nlat_dst,num_nbrs), max_dist

       logical map_found_neighbors(nlon_dst, nlat_dst), src_modulo
       
!-------local variables-----------------------------------------

       integer i, j, n, bound, bound_start, bound_end, i0, j0, 
     .         i_left, i_right, step, step_size, num_neighbors, 
     .         map_dst_xsize, map_dst_ysize, map_src_xsize, 
     .         map_src_ysize, map_dst_size, map_src_size

       real    spherical_distance, d, nearest,res, max_src_dist, pi, 
     .         tpi, hpi, min_theta_dst, max_theta_dst, min_phi_dst, 
     .         max_phi_dst, min_theta_src, max_theta_src, min_phi_src,
     .         max_phi_src, pi180

       logical update_dest_neighbors, continue_search, found_neighbors,
     .         continue_radial_search, result, src_is_modulo


!---------------------------------------------------------------
      

! parameters
      real epsln, large
cc      parameter (epsln=1.e-10, large=1.e20)
      parameter (epsln=4.e-7, large=1.e20)

       pi = 4.*ATAN(1.)
       tpi = 2.0*PI
       hpi = 0.5*PI
       
        pi180 = pi/ 180.

       map_dst_xsize = nlon_dst
       map_dst_ysize = nlat_dst
       map_src_xsize = nlon_src
       map_src_ysize = nlat_src      
       map_dst_size = map_dst_xsize*map_dst_ysize
       map_src_size = map_src_xsize*map_src_ysize
      
       num_neighbors = num_nbrs
       if (num_neighbors .LE. 0) call ef_bail_out (id, 
     .   'num_neighbors must be > 0') 
      
       max_src_dist = max_dist
       src_is_modulo = src_modulo
      
       do i = 1, nlon_dst
       do j = 1, nlat_dst
       do n = 1, num_nbrs
          map_src_add(i,j,n) = 0
          map_src_dist(i,j,n) = large
       enddo
       enddo
       enddo
      
       min_theta_dst=tpi
       max_theta_dst=0.0
       min_phi_dst=pi
       max_phi_dst=-pi
       min_theta_src=tpi
       max_theta_src=0.0
       min_phi_src=pi
       max_phi_src=-pi
      
            
       do i = 1, nlon_dst
       do j = 1, nlat_dst
          if(theta_dst(i,j) .LT. 0.0) theta_dst(i,j)=theta_dst(i,j)+tpi
          if(theta_dst(i,j) .GT. tpi) theta_dst(i,j)=theta_dst(i,j)-tpi
          if(phi_dst(i,j) .LT. -1.* hpi) phi_dst(i,j) = -1.* hpi
          if(phi_dst(i,j) .GT. hpi)  phi_dst(i,j) =  hpi
       enddo
       enddo
       
       do i = 1, map_src_size
          if(theta_src(i) .LT. 0.0)  theta_src(i) = theta_src(i)+tpi
          if(theta_src(i) .GT. tpi)  theta_src(i) = theta_src(i)-tpi
          if(phi_src(i) .LT. -1.* hpi) phi_src(i) = -1.* hpi
          if(phi_src(i) .GT. hpi)  phi_src(i) =  hpi
       enddo

       do j=1,map_dst_ysize
       do i=1,map_dst_xsize
          min_theta_dst = min(min_theta_dst,theta_dst(i,j))
          max_theta_dst = max(max_theta_dst,theta_dst(i,j))
          min_phi_dst = min(min_phi_dst,phi_dst(i,j))
          max_phi_dst = max(max_phi_dst,phi_dst(i,j))
       enddo
       enddo
      
       do i=1,map_src_size
          min_theta_src = min(min_theta_src,theta_src(i))
          max_theta_src = max(max_theta_src,theta_src(i))
          min_phi_src = min(min_phi_src,phi_src(i))
          max_phi_src = max(max_phi_src,phi_src(i))
       enddo
      
!  Note: do comparisons like these with a TM_FPEQ 

c       if (min_phi_dst .LT. min_phi_src) print *, 
c     .    '=> WARNING:  latitute of dest grid exceeds src: src, dest',
c     .    min_phi_src, min_phi_dst, ' degrees src, dest:', 
c     .    min_phi_src/pi180, min_phi_dst/pi180
c       if (max_phi_dst .GT. max_phi_src) print *, 
c     .    '=> WARNING:  latitute of dest grid exceeds src: src, dest',
c     .    max_phi_src, max_phi_dst, ' degrees src, dest:', 
c     .    max_phi_src/pi180, max_phi_dst/pi180
c       if (min_theta_dst .LT. min_theta_src) print *, 
c     .    '=> WARNING : longitude of dest grid exceeds src: src, dest',
c     .    min_theta_src, min_theta_dst, ' degrees src, dest:', 
c     .    min_theta_src/pi180, min_theta_dst/pi180
c       if (max_theta_dst .GT. max_theta_src) print *, 
c     .    '=> WARNING : longitude of dest grid exceeds src: src, dest',
c     .    max_theta_src, max_theta_dst, ' degrees src, dest:', 
c     .    max_theta_src/pi180, max_theta_dst/pi180
      
       do j=1,map_dst_ysize
       do i=1,map_dst_xsize
          found_neighbors=.false.
          continue_search=.true.
          step = 1
          step_size = 1
          nearest = 1.e3
          do while (continue_search .and. step_size .GT. 0)
             do while (step .LE. map_src_size .and. continue_search)
! count land points as nearest neighbors
                d = spherical_distance (theta_dst(i,j), phi_dst(i,j), 
     .                                 theta_src(step),phi_src(step))
                if (d .LE. max_src_dist) then
                   found_neighbors = update_dest_neighbors(
     .                    i, j, map_src_add, map_src_dist, 
     .                    step, d, max_src_dist, 
     .                    nlon_dst, nlat_dst, num_nbrs)
                   if (found_neighbors) then
                      n = 0
                      i0 = mod(step,map_src_xsize)
                      if (i0 .EQ. 0) i0 = map_src_xsize
                      res = float(step)/float(map_src_xsize)
!                     j0 = ceiling(res)
                      j0 = -1.*INT(-1.*res)
                      continue_radial_search = .true.
                      do while (continue_radial_search)
                         n = n+1 ! radial counter
! left boundary 
                         i_left = i0-n
                         if (i_left .LE. 0) then
                            if (src_is_modulo) then
                               i_left = map_src_xsize + i_left
                            else
                               i_left = 1
                            endif
                         endif
                         bound_start = max(j0-n-1,0)*map_src_xsize 
     .                                  + i_left
                         bound_end = min(j0+n-1,map_src_ysize-1)*
     .                                   map_src_xsize + i_left
                         if (bound_end .GT. map_src_size) 
     .                      call ef_bail_out(id, 'array size error')
                         bound = bound_start
                         continue_radial_search = .false.
                         do while (bound .LE. bound_end)
                            d = spherical_distance (theta_dst(i,j),
     .                            phi_dst(i,j), theta_src(bound), 
     .                            phi_src(bound))
                            result = update_dest_neighbors(
     .                        i, j, map_src_add, map_src_dist, 
     .                        bound, d, max_src_dist,
     .                        nlon_dst, nlat_dst, num_nbrs)
                            bound = bound + map_src_xsize
                            if (result) continue_radial_search = .true.
                            if (result) found_neighbors = .true.
                         enddo
! right boundary 
                         i_right = i0+n
                         if (i_right .GT. map_src_xsize) then
                            if (src_is_modulo) then
                               i_right = i_right - map_src_xsize
                            else
                               i_right = map_src_xsize
                            endif
                         endif
                         bound_start = max(j0-n-1,0)*map_src_xsize 
     .                                + i_right
                         bound_end = min(j0+n-1,map_src_ysize-1)*
     .                                 map_src_xsize + i_right
                         bound = bound_start
                         if (bound_end .GT. map_src_size) 
     .                      call ef_bail_out(id, 'array size error')
                         do while (bound .LE. bound_end)
                            d = spherical_distance (theta_dst(i,j),
     .                          phi_dst(i,j),
     .                          theta_src(bound),phi_src(bound))
                            result = update_dest_neighbors(
     .                          i, j, map_src_add, map_src_dist, 
     .                          bound, d, max_src_dist,
     .                          nlon_dst, nlat_dst, num_nbrs)
                            bound = bound + map_src_xsize
                            if (result) continue_radial_search = .true.
                            if (result) found_neighbors = .true.
                         enddo
! bottom boundary 
                         bound_start = max(j0-n-1,0)*map_src_xsize 
     .                                + i_left 
                         bound_end =  max(j0-n-1,0)*map_src_xsize  
     .                                + i_right 
                         if (bound_start .GT. bound_end) then
                            bound_start = max(j0-n-1,0)*map_src_xsize+1
                            bound_end = max(j0-n,1)*map_src_xsize
                         endif
                         bound = bound_start
                         if (bound_end .GT. map_src_size) 
     .                      call ef_bail_out(id, 'array size error')
                         do while (bound .LE. bound_end)
                            d = spherical_distance (theta_dst(i,j),
     .                          phi_dst(i,j),theta_src(bound),
     .                          phi_src(bound))
                            result = update_dest_neighbors(
     .                          i, j, map_src_add, map_src_dist, 
     .                          bound, d, max_src_dist,
     .                          nlon_dst, nlat_dst, num_nbrs)
                            bound = bound + 1
                            if (result) continue_radial_search = .true.
                            if (result) found_neighbors = .true.
                         enddo
! top boundary 
                         bound_start = min(j0+n-1,map_src_ysize-1)*
     .                          map_src_xsize + i_left
                         bound_end =  min(j0+n-1,map_src_ysize-1)*
     .                          map_src_xsize + i_right
                         if (bound_start .GT. bound_end) then
                            bound_start = min(j0+n-1,map_src_ysize-1)*
     .                          map_src_xsize + 1
                            bound_end = min(j0+n,map_src_ysize-1)*
     .                          map_src_xsize
                         endif
                         bound = bound_start
                         if (bound_end .GT. map_src_size) 
     .                      call ef_bail_out(id, 'array size error')
                         do while (bound .LE. bound_end)
                            d = spherical_distance (theta_dst(i,j),
     .                            phi_dst(i,j), theta_src(bound),
     .                            phi_src(bound))
                            result = update_dest_neighbors(
     .                          i, j, map_src_add, map_src_dist, 
     .                          bound, d, max_src_dist,
     .                          nlon_dst, nlat_dst, num_nbrs)
                            bound = bound + 1
                            if (result) continue_radial_search = .true.
                            if (result) found_neighbors = .true.
                         enddo
                      enddo
                      continue_search = .false. ! stop looking
                   endif
                endif
                step=step+step_size
             enddo ! search loop
             step = 1
             step_size = step_size/2
          enddo
          map_found_neighbors(i,j) = found_neighbors
       enddo 
       enddo
      
     .                                
      return
      
      end !  subroutine radial_search
      
!#####################################################################

      real function spherical_distance(theta1,phi1,theta2,phi2)

* Argument definitions
      real theta1, phi1, theta2, phi2

* local variable definitions
      real r1(3), r2(3), cross(3), s, dot, ang, pi

! parameters
      real epsln
cc      parameter (epsln=1.e-10)
      parameter (epsln=4.e-7)

! this is a simple, enough way to calculate distance on the sphere
! first, construct cartesian vectors r1 and r2
! then calculate the cross-product which is proportional to the area
! between the 2 vectors.  The angular distance is arcsin of the 
! distancealong the sphere
!
! theta is longitude and phi is latitude
!

      pi =  4.*ATAN(1.)

      r1(1) = cos(theta1)*cos(phi1)
      r1(2)=sin(theta1)*cos(phi1)
      r1(3)=sin(phi1)

      r2(1) = cos(theta2)*cos(phi2)
      r2(2)=sin(theta2)*cos(phi2)
      r2(3)=sin(phi2)

      cross(1) = r1(2)*r2(3)-r1(3)*r2(2)
      cross(2) = r1(3)*r2(1)-r1(1)*r2(3)
      cross(3) = r1(1)*r2(2)-r1(2)*r2(1)

      s = sqrt(cross(1)**2.+cross(2)**2.+cross(3)**2.)

      s = min(s,1.0-epsln)

      dot = r1(1)*r2(1) + r1(2)*r2(2) + r1(3)*r2(3)

      if (dot .GT. 0) then
          ang = asin(s)
      else if (dot .LT. 0) then
          ang = pi - asin(s)
      else
          ang = pi/2.
      endif
      
      spherical_distance = abs(ang) ! in radians
      
      return
      
      end ! function spherical_distance

!#####################################################################
      logical function update_dest_neighbors(i, j, map_src_add, 
     .             map_src_dist, src_add, d, max_src_dist, nx, ny, nn)
 
* Argument definitions
       integer i, j, nx, ny, nn, map_src_add(nx, ny, nn), src_add
       real map_src_dist(nx, ny, nn), d, max_src_dist

* local variables
      integer n,m, num_neighbors

      update_dest_neighbors = .false.
      num_neighbors = nn

! if (d .le. max_src_dist) then
!    NLOOP : do n=1,num_neighbors
!       DIST_CHK : if (d .lt. map_src_dist(n)) then
!          if (n > 1 .and. src_add == map_src_add(n-1)) exit NLOOP
!          do m=num_neighbors,n+1,-1
!             map_src_add(m) = map_src_add(m-1)
!             map_src_dist(m) = map_src_dist(m-1)
!          enddo
!          map_src_add(n) = src_add
!          map_src_dist(n) = d
!          update_dest_neighbors = .true.
!          exit NLOOP ! n loop
!       endif DIST_CHK
!   end do NLOOP
! endif

      if (d .le. max_src_dist) then
         do n=1,num_neighbors 
            if (d .lt. map_src_dist(i,j,n)) then 
cc               if (n .GT. 1 .and. src_add .NE. map_src_add(i,j,n-1)) 
ccc               if (n .GT. 1 .and. src_add .EQ. map_src_add(i,j,n-1)) 
ccc     .             GOTO 100
               do m=num_neighbors,n+1,-1
                  map_src_add(i,j,m) = map_src_add(i,j,m-1)
                  map_src_dist(i,j,m) = map_src_dist(i,j,m-1)
               enddo
               map_src_add(i,j,n) = src_add
               map_src_dist(i,j,n) = d
               update_dest_neighbors = .true.
               goto 100
            endif 
         end do 
  100    continue

      endif

      return
      end ! function update_dest_neighbors



!#######################################################################
            
       subroutine save_mapping (src_dist, wt, mask_dst, 
     .      found_neighbors, i_lon, j_lat, max_src_dist,  
     .      nlon_src, nlat_src, nlon_dst, nlat_dst, num_nbrs,
     .      bad_flag_result, result)

! compute mask_dst, wt, i_lon, and j_lat and save in the result.
! result(nlon_dst,nlat_dst,num_nbrs,2)  where num_nbrs = 4 and
! 
!   result(m,n,1,1) = mask_dst(m,n)
!   result(m,n,k,2) = wt(m,n,k)
!   result(m,n,k,3) = i_lon(m,n,k)
!   result(m,n,k,4) = j_lat(m,n,k)


       integer nlon_src, nlat_src, nlon_dst, nlat_dst, num_nbrs

       real src_dist(nlon_dst, nlat_dst, num_nbrs)
       real wt(nlon_dst, nlat_dst, num_nbrs)  
       real mask_dst(nlon_dst, nlat_dst)  
       real max_src_dist, bad_flag_result

       integer i_lon(nlon_dst,nlat_dst,num_nbrs)
       integer j_lat(nlon_dst,nlat_dst,num_nbrs) 

! mask, wt, etc for applying the regridding
       real result(nlon_dst, nlat_dst, num_nbrs, 4)  

       logical found_neighbors(nlon_dst,nlat_dst)

!--- some local variables ----------------------------------------
       integer nlon_in, nlat_in, nlon_out, nlat_out, num_neighbors,
     .               m, n, k, i1, i2, j1, j2
       real sum
!-----------------------------------------------------------------
      
      real epsln, large
cc      parameter (epsln=1.e-10, large=1.e20)
      parameter (epsln=4.e-7, large=1.e20)

      result(1,1,2,1) = nlon_src
      result(1,1,3,1) = nlat_src
      
         nlon_in  = nlon_src
         nlat_in  = nlat_src
         nlon_out = nlon_dst
         nlat_out = nlat_dst   
         num_neighbors = num_nbrs

!first calculate destination mask and weight
      
         do m=1,nlon_out
         do n=1,nlat_out
            mask_dst(m,n) = 1.0
            IF (.not. found_neighbors(m,n)) mask_dst(m,n) = 0.0

! neighbors are sorted nearest to farthest
! check nearest to see if it is a land point

            i1 = i_lon(m,n,1)
            j1 = j_lat(m,n,1)
      
            if( i1 .eq. 0 .and. j1 .eq. 0) then
               mask_dst(m,n) = 0.0
!            else if (mask_src(i1,j1) .lt. 0.5) then
!               mask_dst(m,n) = 0.0
            endif
            
            if(num_neighbors .gt. 1 ) then
               i2 = i_lon(m,n,2)
               j2 = j_lat(m,n,2)

! compare first 2 nearest neighbors -- if they are nearly
! equidistant then use this mask for robustness

!               if(i2 .ne. 0 .and. j2 .ne. 0 .and. 
!     .            abs(src_dist(m,n,2)-src_dist(m,n,1)) .lt. epsln) then
!                  if((mask_src(i1,j1) .lt. 0.5))  mask_dst(m,n) = 0.0
!               endif
            endif
      
            sum=0.0
            do k=1, num_neighbors
              if (src_dist(m,n,k) .LE. epsln) then
                 wt(m,n,k) = large
                 sum = sum + large
              else if(src_dist(m,n,k) .LE. max_src_dist ) then
                 wt(m,n,k) = 1.0/src_dist(m,n,k)
                 sum = sum+wt(m,n,k)
              else
                 wt(m,n,k) = 0.0
              endif
           enddo
           if (sum .GT. epsln) then
              do k = 1, num_neighbors
                 wt(m,n,k) = wt(m,n,k)/sum
              enddo
           else
              mask_dst(m,n) = 0.0
           endif
      
         enddo
         enddo
      
! Save mask_dst, wt, i_lon, j_lat for this curvi/rect grid pair.

         do m=1,nlon_out
         do n=1,nlat_out
            result(m,n,1,1) = mask_dst(m,n)
            do k=1, num_neighbors
               result(m,n,k,2) = wt(m,n,k)
               result(m,n,k,3) = i_lon(m,n,k)
               result(m,n,k,4) = j_lat(m,n,k)
            enddo
         enddo
         enddo
      
         return
       end ! subroutine save_mapping
