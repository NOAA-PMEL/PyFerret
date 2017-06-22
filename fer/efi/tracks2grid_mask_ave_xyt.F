*
*  tracks2grid_mask_ave_xyt.F
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
* June 2009  from scatgrid_nobs_xyt
*
* 9/ 3/2015 Fix LAS/SOCAT ticket 1720, missingness patterns in cruise-weighted 
*           vs unweighted gridded fields.
* 1/27/2016 Apply modulo operation to scattered longitudes or time if the output
*           axis is a modulo axis.
*
* The input data points are not a simple scatter.  They are cruise tracks 
* strung one after another .  (Each track will be a contiguous block of points 
* in the input data arrays. )  And the subtlety is that each cruise needs to 
* receive the same weight when determining the average for a cell, rather than 
* each individual data point.  Which means effectively that you need to have 
* two averaging processes .  You need to have a buffer (the individual cruise 
* buffer) that is the size and shape of the output grid in which you average 
* each individual cruise.  And then when the final point of each cruise from 
* the input data is encountered, you need to add (average) the points from the 
* individual cruise buffer into the output buffer, re-zeroing the individual 
* cruise buffer as you do so, so it is ready to average the next cruise. 
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


      SUBROUTINE  tracks2grid_mask_ave_xyt_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Average by cruise, when gridding scattered ',
     .        'data to XYT grid where output grid is masked.')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 9)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS,
     .        IMPLIED_BY_ARGS, NORMAL, IMPLIED_BY_ARGS)

      CALL ef_set_num_work_arrays(id, 8)

* Output grid is determined by arguments 4, 5, 6, the result's x, y, t axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'VAR')
      CALL ef_set_arg_desc(id, arg, 
     .     'Variable at scattered input XYT. Enter a scalar to just count Obs.')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'X coordinates of scattered input triples')

      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Y coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'T coordinates of scattered input triples')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'ICRUISE')
      CALL ef_set_arg_desc(id, arg, 
     .     'Cruise ID at scattered input locations')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     . 'X axis coordinates of the output grid, with same units as xpts')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'YAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     . 'Y axis coordinates of the output grid, with same units as ypts')
      CALL ef_set_axis_influence(id, arg, NO, YES, NO, NO)

      arg = 8
      CALL ef_set_arg_name(id, arg, 'TAXPTS')
      WRITE (buff, 20)
   20 FORMAT ('T axis coordinates, of the output grid, with same ',
     .       'units and origin as tpts')
      CALL ef_set_arg_desc(id, arg, buff)
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

      arg = 9
      CALL ef_set_arg_name(id, arg, 'MASK')
      CALL ef_set_arg_desc(id, arg,  
     .                   'XY mask on the output grid 1=valid, 0=ignore')
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, NO)

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
      SUBROUTINE tracks2grid_mask_ave_xyt_work_size(id)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER nxout, nyout, ntout, nx2, ny2, nt2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG6) - arg_lo_ss(X_AXIS,ARG6)
      nyout = 1 + arg_hi_ss(Y_AXIS,ARG7) - arg_lo_ss(Y_AXIS,ARG7)
      ntout = 1 + arg_hi_ss(T_AXIS,ARG8) - arg_lo_ss(T_AXIS,ARG8)

* xaxlo  output x axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nxout, 1, 1, 1)
* xaxhi  output x axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, nxout, 1, 1, 1)

* yaxlo  output y axis
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, nyout, 1, 1, 1)
* yaxhi  output y axis
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, nyout, 1, 1, 1)

* taxlo  output t axis
      CALL ef_set_work_array_dims (id, 5, 1, 1, 1, 1, ntout, 1, 1, 1)
* taxhi  output t axis
      CALL ef_set_work_array_dims (id, 6, 1, 1, 1, 1, ntout, 1, 1, 1)

* obs_cruise  work array - sum and counts of observations per cruise
* x,y,t direction: output grid
* z direction: 1..2
      CALL ef_set_work_array_dims (id, 7, 1, 1, 1, 1, 
     .                             nxout, nyout, 2, ntout)

* obs  work array - work array - sum and counts of observations
* x,y,t direction: output grid
* z direction: 1..2
      CALL ef_set_work_array_dims (id, 8, 1, 1, 1, 1, 
     .                             nxout, nyout, 2, ntout)

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE tracks2grid_mask_ave_xyt_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, result, 
     .     xaxlo, xaxhi, yaxlo, yaxhi, taxlo, taxhi, obs_cruise, obs)

*  arg_1  var
*  arg_2  xpts  \
*  arg_3  ypts   | Scattered x,y triples to be gridded.
*  arg_4  tpts  / 
*  arg_5  icruise
*
*  arg_6  xaxis of new grid   
*  arg_7  yaxis of new grid
*  arg_8  taxis of new grid
*  arg_9  xy mask for output grid

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .     mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .     mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .     mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, 
     .     mem4loz:mem4hiz, mem4lot:mem4hit)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, 
     .     mem5loz:mem5hiz, mem5lot:mem5hit)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, 
     .     mem6loz:mem6hiz, mem6lot:mem6hit)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, 
     .     mem7loz:mem7hiz, mem7lot:mem7hit)
      REAL arg_8(mem8lox:mem8hix, mem8loy:mem8hiy, 
     .     mem8loz:mem8hiz, mem8lot:mem8hit)
      REAL arg_9(mem9lox:mem9hix, mem9loy:mem9hiy, 
     .     mem9loz:mem9hiz, mem9lot:mem9hit)

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
      INTEGER i1, i2, j1, k1, l1, k2
      INTEGER i6,j7,L8,i6n,j7n,L8n

      INTEGER nxpts, npts
      INTEGER nx, ny, nt
      INTEGER i1n, i2n
      INTEGER obs_only, ncc, nscat

      REAL x1, y1, t1, icruise, ncruise, xf, yf, tf
      REAL xx, yy, tt, vv

*  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), mdulo(6), regular(6)
      LOGICAL modulox, modulot
      REAL*8 modxlen, modtlen

*  Dimension the work arrays

      REAL*8 xaxlo(wrk1lox:wrk1hix)
      REAL*8 xaxhi(wrk2lox:wrk2hix)
      REAL*8 yaxlo(wrk3lox:wrk3hix)
      REAL*8 yaxhi(wrk4lox:wrk4hix)
      REAL*8 taxlo(wrk5lox:wrk5hix)
      REAL*8 taxhi(wrk6lox:wrk6hix)

      REAL obs_cruise(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .               wrk7loz:wrk7hiz, wrk7lot:wrk7hit)
      REAL obs(wrk8lox:wrk8hix, wrk8loy:wrk8hiy,
     .               wrk8loz:wrk8hiz, wrk8lot:wrk8hit)

      CHARACTER*250 errtxt

      REAL*8 dx, dy, dt, xxbeg, xxend, yybeg, yyend
      REAL*8 xax1,xaxf, yax1,yaxf, tax1,taxf
      INTEGER indx(4,2)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


*  Check to see if the output X-axis is modulo (does not need to be regular)
      CALL ef_get_axis_info_6d(id, ARG6, ax_name, ax_units,
     .                         backward, mdulo, regular)
      modulox = mdulo(X_AXIS)
      IF ( modulox ) THEN
         CALL ef_get_axis_modulo_len(id, ARG6, X_AXIS, modxlen)
      ENDIF

*  Check to see if the output T-axis is modulo (does not need to be regular)
      CALL ef_get_axis_info_6d(id, ARG6, ax_name, ax_units,
     .                         backward, mdulo, regular)
      modulot = mdulo(T_AXIS)
      IF ( modulot ) THEN
         CALL ef_get_axis_modulo_len(id, ARG8, T_AXIS, modtlen)
      ENDIF

      obs_only = 0	! 0 = var 1 is a variable to average.
                        ! 1 = Compute a count of cruises with an obs in the grid cells

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, Z, or T axis of the input arguments.

      nxpts = 0

      DO 100 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG2) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG2)
            i1n = arg_hi_ss(m,ARG2)
            if (i1n-i1 .NE. 0) nxpts = 1 + (i1n - i1)
            IF (nxpts .EQ. 0) GOTO 900
         ENDIF
 100  CONTINUE

      npts = 1
      DO 110 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG1)
            i2n = arg_hi_ss(m,ARG1)
            npts = 1 + (i2n - i2)
            IF ((npts.NE.1) .AND. (npts.NE.nxpts)) GOTO 900
         ENDIF
 110  CONTINUE
      IF (npts .EQ. 1) obs_only = 1  ! just count up the obs

      DO 120 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG3) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG3)
            i2n = arg_hi_ss(m,ARG3)
            npts = 1 + (i2n - i2)
            IF (npts .NE. nxpts) GOTO 900
         ENDIF
 120  CONTINUE

      DO 130 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG4) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG4)
            i2n = arg_hi_ss(m,ARG4)
            npts = 1 + (i2n - i2)
            IF (npts .NE. nxpts) GOTO 900
         ENDIF
 130  CONTINUE

      DO 140 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG5) .GE. 1) THEN
            i2 = arg_lo_ss(m,ARG5)
            i2n = arg_hi_ss(m,ARG5)
            npts = 1 + (i2n - i2)
            IF (npts .NE. nxpts) GOTO 900
         ENDIF
 140  CONTINUE

      nscat = nxpts

*  Compute number of points in output axes.

      i6 = ef_unspecified_int4
      j7 = ef_unspecified_int4
      L8 = ef_unspecified_int4

      i6 = arg_lo_ss(X_AXIS,ARG6)
      i6n = arg_hi_ss(X_AXIS,ARG6)

      j7 = arg_lo_ss(Y_AXIS,ARG7)
      j7n = arg_hi_ss(Y_AXIS,ARG7)

      L8 = arg_lo_ss(T_AXIS,ARG8)
      L8n = arg_hi_ss(T_AXIS,ARG8)

      nx = 1 + (i6n - i6)
      ny = 1 + (j7n - j7)
      nt = 1 + (L8n - L8)

*  Check that xax is a X axis and yax a Y axis

      IF (i6 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Sixth argument must be a X axis'
         GO TO 999
      ENDIF
      IF (j7 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Seventh argument must be a Y axis'
         GO TO 999
      ENDIF
      IF (L8 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Eighth argument must be a T axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axes.

      CALL EF_GET_BOX_LIMITS(id, ARG6, X_AXIS,
     .   arg_lo_ss(X_AXIS, ARG6), arg_hi_ss(X_AXIS, ARG6), xaxlo, xaxhi)

      CALL EF_GET_BOX_LIMITS(id, ARG7, Y_AXIS,
     .   arg_lo_ss(Y_AXIS, ARG7), arg_hi_ss(Y_AXIS, ARG7), yaxlo, yaxhi)

      CALL EF_GET_BOX_LIMITS(id, ARG8, T_AXIS,
     .   arg_lo_ss(T_AXIS, ARG8), arg_hi_ss(T_AXIS, ARG8), taxlo, taxhi)

*  Get first coordinates of output axes.

      call ef_get_coordinates(id, ARG6, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG6), arg_lo_ss(X_AXIS, ARG6), xax1)
      call ef_get_coordinates(id, ARG7, X_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG7), arg_lo_ss(Y_AXIS, ARG7), yax1)
      x1 = xax1
      y1 = yax1

*  Compute result

*  Initialize counts.

      k2 = 2
      IF (obs_only .EQ. 1) k2 = 1

      DO L = 1, nt
      DO k = 1, k2
      DO j = 1, ny
      DO i = 1, nx
         obs(i,j,k,L) = 0.
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      DO L = 1, nt
      DO k = 1, k2
      DO j = 1, ny
      DO i = 1, nx
         obs_cruise(i,j,k,L) = 0.
      ENDDO
      ENDDO
      ENDDO
      ENDDO

*  Loop over x and y, compute the number of observations for the gaussian-weighted 
*  mapping onto the grid.  Lat and longitude may be on the X,Y,Z or T axis of ARG1 
*  and ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

      n = 1
      CALL pickoutxytcv (obs_only, arg_1, arg_2, arg_3, arg_4, arg_5,
     .           n, vv, xx, yy, tt, ncc)
      icruise = ncc

*  If output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
         IF ( modulox ) THEN
            CALL modscat(xaxlo(1), xaxhi(nx), modxlen, 1, xx)
         ENDIF
         IF ( modulot ) THEN
            CALL modscat(taxlo(1), taxhi(nt), modtlen, 1, tt)
         ENDIF


* For each cruise keep track of the index range it spans in the result grid
* Then at the end of the cruise need only to process that subset of the grid.
      indx(1,1) = nx
      indx(1,2) = 1
      indx(2,1) = ny
      indx(2,2) = 1
      indx(4,1) = nt
      indx(4,2) = 1
      
      DO 300 n = 1, nscat

         CALL pickoutxytcv (obs_only, arg_1, arg_2, arg_3, arg_4, arg_5,
     .           n, vv, xx, yy, tt, ncc)
         IF (vv .EQ. bad_flag(ARG1) .OR. xx .EQ. bad_flag(ARG2) .OR. 
     .          yy .EQ. bad_flag(ARG3) .OR. tt .EQ. bad_flag(ARG4) ) GOTO 300
     


*  If output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
         IF ( modulox ) THEN
            CALL modscat(xaxlo(1), xaxhi(nx), modxlen, 1, xx)
         ENDIF
         IF ( modulot ) THEN
            CALL modscat(taxlo(1), taxhi(nt), modtlen, 1, tt)
         ENDIF


	 if (xx.LT.xaxlo(1) .OR. xx.GT.xaxhi(nx)) GOTO 300
	 if (yy.LT.yaxlo(1) .OR. yy.GT.yaxhi(ny)) GOTO 300
	 if (tt.LT.taxlo(1) .OR. tt.GT.taxhi(nt)) GOTO 300

         IF (ncc .NE. icruise) THEN
	    
            DO L = indx(4,1), indx(4,2)
            DO j = indx(2,1), indx(2,2)
            DO i = indx(1,1), indx(1,2)
	       IF (obs_only .EQ. 0) THEN
                  IF (obs_cruise(i,j,2,L) .GT. 0.) THEN
                     obs(i,j,1,L) = obs(i,j,1,L) + 
     .                 obs_cruise(i,j,1,L)/obs_cruise(i,j,2,L)
                     obs(i,j,2,L) = obs(i,j,2,L) + 1.
		  ENDIF
                  obs_cruise(i,j,1,L) = 0.
                  obs_cruise(i,j,2,L) = 0.
               ELSE
                  IF (obs_cruise(i,j,1,L) .GT. 0.) THEN
                     obs(i,j,1,L) = obs(i,j,1,L) + 1.
                  ENDIF
                  obs_cruise(i,j,1,L) = 0.
               ENDIF
            ENDDO
            ENDDO
            ENDDO

	    icruise = ncc

* reset indices
            indx(1,1) = nx
            indx(1,2) = 1
            indx(2,1) = ny
            indx(2,2) = 1
            indx(4,1) = nt
            indx(4,2) = 1
	    
	 ENDIF

         CALL nobsxytv (obs_only, xx, yy, tt, vv, nx, ny, nt, 
     .           xaxlo,xaxhi, yaxlo,yaxhi, taxlo,taxhi, indx, obs_cruise)

 300     CONTINUE

* Get data from the last cruise
	    
      DO L = indx(4,1), indx(4,2)
      DO j = indx(2,1), indx(2,2)
      DO i = indx(1,1), indx(1,2)
         IF (obs_only .EQ. 0) THEN
            IF (obs_cruise(i,j,2,L) .GT. 0.) THEN
               obs(i,j,1,L) = obs(i,j,1,L) + 
     .                 obs_cruise(i,j,1,L)/obs_cruise(i,j,2,L)
               obs(i,j,2,L) = obs(i,j,2,L) + 1.
            ENDIF
            obs_cruise(i,j,1,L) = 0.
            obs_cruise(i,j,2,L) = 0.
         ELSE
            IF (obs_cruise(i,j,1,L) .GT. 0.) THEN
               obs(i,j,1,L) = obs(i,j,1,L) + 1.
            ENDIF
               obs_cruise(i,j,1,L) = 0.
         ENDIF
      ENDDO
      ENDDO
      ENDDO

*  Put gridded fcn into result variable.

      IF (obs_only .EQ. 0) THEN
         k = res_lo_ss(Z_AXIS)
         L1 = 1
         DO 420 L = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            j1 = 1
            DO 410 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
               i1 = 1
               DO 400 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
                  IF (obs(i1,j1,2,L1) .GT. 0.) THEN
                     result(i,j,k,l) = obs(i1,j1,1,L1)/obs(i1,j1,2,L1)
		  ELSE
                     result(i,j,k,l) = bad_flag_result
		  ENDIF

                  i1 = i1 + 1
 400           CONTINUE
               j1 = j1 + 1

 410        CONTINUE
            L1 = L1 + 1
 420     CONTINUE

      ELSE
         k = res_lo_ss(Z_AXIS)
         L1 = 1
         DO 520 L = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            j1 = 1
            DO 510 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
               i1 = 1
               DO 500 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
                  result(i,j,k,l) = obs(i1,j1,1,L1)
                  i1 = i1 + 1
 500           CONTINUE
               j1 = j1 + 1

 510        CONTINUE
            l1 = l1 + 1
 520     CONTINUE

      ENDIF

      RETURN

900   CONTINUE

      IF (nxpts .NE. npts) THEN
         WRITE (errtxt,20)
      ELSE IF (nxpts .EQ. 0) THEN
         WRITE (errtxt, 30)
      ENDIF
      GOTO 999 

 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN

   20 FORMAT 
     . ('Input scattered var,x,y,t,cruiseID have different # of points')

   30 FORMAT ('No data in scattered x,y,t points')


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      END



