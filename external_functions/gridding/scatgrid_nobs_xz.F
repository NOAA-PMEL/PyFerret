*
*  scatgrid_nobs_xz.F
*
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
* Ansley Manke
* February 9, 2005 

* Returns # observations in each grid cell when a variable is interpolated 
* onto an equally-spaced X-Z grid.  Input is scattered locations xpts, zpts
* and the desired result grid. Output is # observations in each grid cell 
* when data is gridded in x and y.  Calls routine "nobs" based on Billy
* Kessler's gaussmap-subs.f.
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
* axis_influence     Are this argument's axes the same as the result grid?
*                       ( YES, NO )
*
* axis_extend       How much does Ferret need to extend arg limits relative to result 
*


      SUBROUTINE  scatgrid_nobs_xz_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Count observations when gridding scattered ',
     .        'data to XZ grid.')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 4)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .      NORMAL, IMPLIED_BY_ARGS, NORMAL)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 3)

* Output grid is determined by arguments 3 and 4, the result's x and y axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'X coordinates of scattered input pairs')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'ZPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Z coordinates of scattered input pairs')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'X axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'ZAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'Z axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN 
      END
*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE scatgrid_nobs_xz_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER nxout, nzout, nx2, nz2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3)
      nzout = 1 + arg_hi_ss(Z_AXIS,ARG4) - arg_lo_ss(Z_AXIS,ARG4)

      nx2 = nxout* 2
      nz2 = nzout* 2

* xax  output x axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nx2, 1, 1, 1)

* zax  output z axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, nz2, 1, 1, 1)

* obs  work array - counts of observations 
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, 
     .                             nxout, nzout, 1, 1)

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE  scatgrid_nobs_xz_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, result, xax, zax, obs)

*  arg_1  xpts  \Scattered x,y pairs to be gridded.
*  arg_2  zpts  / 
*  arg_3  xaxis of new grid   
*  arg_4  zaxis of new grid

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id
      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .     mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .     mem4lot:mem4hit)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      INTEGER i, j, k, l, m, n
      INTEGER i1, i2, j1, k1, l1
      INTEGER i3, i3n, k4, k4n

      INTEGER nxpts, nzpts, nscat
      INTEGER nx, nz    
      INTEGER i1n, i2n

      REAL x1, z1, xf, zf
      REAL xx, zz

*  Dimension the work arrays

      REAL*8 xax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 zax(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL obs(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)

      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulox(4), moduloz(4), regular(4)

      REAL dx, dz, xxbeg, xxend, zzbeg, zzend


      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


*  Check to see if output axes are modulo

      CALL ef_get_axis_info (id, 3, ax_name, ax_units, backward, 
     .                       modulox, regular)
      CALL ef_get_axis_info (id, 4, ax_name, ax_units, backward, 
     .                       moduloz, regular)

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, Z, or T axis of the input arguments.

      nxpts = 0
      nzpts = 0

      DO 100 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG1)
            i1n = arg_hi_ss(m,ARG1)
            if (i1n-i1 .NE. 0) nxpts = 1 + (i1n - i1)
         ENDIF
 100  CONTINUE

      DO 110 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG2)
            i2n = arg_hi_ss(m,ARG2)
            if (i2n-i2 .NE. 0) nzpts = 1 + (i2n - i2)
         ENDIF
 110  CONTINUE

      IF (nxpts .NE. nzpts .OR. nxpts .EQ. 0) GOTO 900

      nscat = nxpts

*  Compute number of points in output axes.

      i3 = arg_lo_ss(X_AXIS,ARG3)
      i3n = arg_hi_ss(X_AXIS,ARG3)

      k4 = arg_lo_ss(Z_AXIS,ARG4)
      k4n = arg_hi_ss(Z_AXIS,ARG4)

      nx = 1 + (i3n - i3)
      nz = 1 + (k4n - k4)

*  Check that xax is a X axis and zax a Z axis

      IF (i3 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Third argument must be a X axis'
         GO TO 999
      ENDIF
      IF (k4 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Fourth argument must be a Z axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axes.

      call ef_get_coordinates(id, ARG3, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG3), arg_hi_ss(X_AXIS, ARG3), xax)

      call ef_get_coordinates(id, ARG4, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, ARG4), arg_hi_ss(Z_AXIS, ARG4), zax)

*  Set start, end, and delta for output axes. 

      x1 = xax(1,1,1,1)
      z1 = zax(1,1,1,1)

      xf = xax(nx,1,1,1)
      zf = zax(nz,1,1,1)

*  Gridbox sizes in data units
      dx = xf - x1
      IF (nx .GT. 1) dx = (xf-x1)/ REAL(nx-1)
      dz = zf - z1
      IF (nz .GT. 1) dz = (zf-z1)/ REAL(nz-1) 


*  Compute result

*  Initialize counts.

         DO j = 1, nz
            DO i = 1, nx
               obs(i,j,1,1) = 0.
            ENDDO
         ENDDO

*  Loop over x and z, compute the number of observations for the gaussian-weighted 
*  mapping onto the grid.  Lat and longitude may be on the X,Y,Z or T axis of ARG1 
*  and ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

         DO 300 n = 1, nscat

            CALL pickoutxyz (arg_1, arg_2, n, xx, zz)

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.

            IF (modulox(1)) CALL modscat (xax, nx, 1, xx)
            IF (moduloz(2)) CALL modscat (zax, nz, 1, zz)

	    CALL nobs (xx, zz, nx, nz, x1, z1, dx, dz,
     .		       xf, zf, obs)

C ACM modulo 11/9/00  Put points within dx or dz of the end just beyond the
C                     other end, and use in the gridding computation.

            IF (modulox(1)) THEN
               IF ((xx-x1 .GE. 0.) .AND. (xx-x1 .LT. dx) ) THEN
                  xxend = xf + (xx-x1)
                  CALL nobs (xxend, zz, nx, nz, x1, z1, dx, dz,
     .                       xf, zf, obs)
               ENDIF
                     
               IF ((xf-xx .GE. 0.)  .AND. (xf-xx .LT. dx) ) THEN
                  xxbeg = x1 - (xf-xx)
                  CALL nobs (xxbeg, zz, nx, nz, x1, z1, dx, dz,
     .                       xf, zf, obs)
               ENDIF
            ENDIF

            IF (moduloz(2)) THEN
               IF ((zz-z1 .GE. 0.0) .AND.  (zz-z1  .LT. dz) ) THEN
                  zzend = zf + (zz-z1)
                  CALL nobs (xx, zzend, nx, nz, x1, z1, dx, dz,
     .                       xf, zf, obs)
               ENDIF
                     
               IF ((zf-zz .LT. 0.)  .AND.  (zf-zz .LT. dz) ) THEN
                  zzbeg = z1 - (zf-zz)
                  CALL nobs (xx, zzbeg, nx, nz, x1, z1, dx, dz,
     .		             xf, zf, obs)
               ENDIF
            ENDIF


 300     CONTINUE

*  Put gridded fcn into result variable.

         j = res_lo_ss(Y_AXIS)
         l = res_lo_ss(T_AXIS)

         i1 = 1
         DO 410 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
            k1 = 1
            DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               result(i,j,k,l) = obs(i1,k1,1,1)
               k1 = k1 + 1

 400        CONTINUE
            i1 = i1 + 1
 410     CONTINUE

      RETURN

900   CONTINUE

      IF (nxpts .NE. nzpts) THEN
         WRITE (errtxt,20) nxpts, nzpts
      ELSE IF (nxpts .EQ. 0) THEN
         WRITE (errtxt, 30)
      ENDIF
      GOTO 999 

 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN


   20 FORMAT ('Input scattered x, z have different # of points', 2I8)

   30 FORMAT ('No data in scattered x, z points')


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      END
