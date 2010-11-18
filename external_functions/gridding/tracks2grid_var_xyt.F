*
*  tracks2grid_var_xyt.F
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
* October 2010 from tracks2grid_ave_xyt.F
*
* This is the description of the averaging function.
* The input data points are not a simple scatter.  They are cruise tracks 
* strung one after another.  (Each track will be a contiguous block of points 
* in the input data arrays.)  And the subtlety is that each cruise needs to 
* receive the same weight when determining the average for a cell, rather than 
* each individual data point.  Which means effectively that you need to have 
* two averaging processes.  You need to have a buffer (the individual cruise 
* buffer) that is the size and shape of the output grid in which you average 
* each individual cruise.  And then when the final point of each cruise from 
* the input data is encountered, you need to add (average) the points from the 
* individual cruise buffer into the output buffer, re-zeroing the individual 
* cruise buffer as you do so, so it is ready to average the next cruise. 
*
* This function will return the variance for the averages, based on weighted means,
* for small samples, http://wapedia.mobi/en/Weighted_mean#7.
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


      SUBROUTINE  tracks2grid_var_xyt_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Average by cruise, when gridding scattered ',
     .        'data to XYT grid.')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 8)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS,
     .        IMPLIED_BY_ARGS, NORMAL, IMPLIED_BY_ARGS)

      CALL ef_set_num_work_arrays(id, 9)

* Output grid is determined by arguments 4, 5, 6, the result's x, y, t axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'VAR')
      CALL ef_set_arg_desc(id, arg, 'Variable at scattered input XYT.')
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
     .                   'X axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'YAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'Y axis coordinates of a regular output grid')
      CALL ef_set_axis_influence(id, arg, NO, YES, NO, NO)

      arg = 8
      CALL ef_set_arg_name(id, arg, 'TAXPTS')
      WRITE (buff, 20)
   20 FORMAT ('T axis coordinates, of a regular output grid, with same ',
     .       'units and origin as tpts')
      CALL ef_set_arg_desc(id, arg, buff)
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)



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
      SUBROUTINE tracks2grid_var_xyt_work_size(id)

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
      INTEGER nxout, nyout, ntout, nx2, ny2, nt2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG6) - arg_lo_ss(X_AXIS,ARG6)
      nyout = 1 + arg_hi_ss(Y_AXIS,ARG7) - arg_lo_ss(Y_AXIS,ARG7)
      ntout = 1 + arg_hi_ss(T_AXIS,ARG8) - arg_lo_ss(T_AXIS,ARG8)

      nx2 = nxout* 2
      ny2 = nyout* 2
      nt2 = ntout* 2

* xax  output x axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nx2, 1, 1, 1)

* yax  output y axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, ny2, 1, 1, 1)

* Tax  output t axis
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, nt2, 1, 1, 1)

* obs_cruise  work array - sum and counts of observations per cruise
* x,y,t direction: output grid
* z direction: 1..2
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, 
     .                             nxout, nyout, 2, ntout)

* cell_avg  work array - work array - sum and counts over cruises for cell average
* x,y,t direction: output grid
* z direction: 1..2
      CALL ef_set_work_array_dims (id, 5, 1, 1, 1, 1, 
     .                             nxout, nyout, 2, ntout)

* wi_sums k=1, sum of weights wi = sum of obs in the cells
* x,y,t direction: output grid
* z direction: 1
      CALL ef_set_work_array_dims (id, 6, 1, 1, 1, 1, 
     .                             nxout, nyout, 1, ntout)

* cruisespan getting the min and max lon/lat for computing
* the span of a cruise in each cell. 
      CALL ef_set_work_array_dims (id, 7, 1, 1, 1, 1, 
     .                             nxout, nyout, 4, ntout)

* cruisemean saving individual weighted cruise means. 
* for computing s^2  sum of wi*(xi - weighted_mean)**2
* x,y,t direction: output grid
* z direction: 30 (max cruises per cell?)
*   There are more efficient ways to do this, but this is 
*   a clean way.
      CALL ef_set_work_array_dims (id, 8, 1, 1, 1, 1, 
     .                             nxout, nyout, 30, ntout)

* span
* for arc length for cruises in grid cell.
* x,y,t direction: output grid
* z direction: 0 (max cruises per cell?)
      CALL ef_set_work_array_dims (id, 9, 1, 1, 1, 1, 
     .                             nxout, nyout, 30, ntout)

      RETURN
      END

*
* In this subroutine we compute the result
*

      SUBROUTINE tracks2grid_var_xyt_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, arg_5, arg_6, arg_7, arg_8, result, xax, yax, tax, 
     .     obs_cruise, cell_avg, wi_sums, cruisespan, cruisemean, span)

*  arg_1  var
*  arg_2  xpts  \
*  arg_3  ypts   | Scattered x,y triples to be gridded.
*  arg_4  tpts  / 
*  arg_5  icruise
*
*  arg_6  xaxis of new grid   
*  arg_7  yaxis of new grid
*  arg_8  taxis of new grid

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

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
      INTEGER i6,j7,L8,i6n,j7n,L8n

      INTEGER nxpts, npts
      INTEGER nx, ny, nt
      INTEGER i1, i2, i1n, i2n
      INTEGER obs_only, icruise, ncc, nc_cell, nk

      REAL x1, y1, t1, xf, yf, tf
      REAL xx, yy, tt, vv


      REAL dx, dy, xxbeg, xxend, yybeg, yyend
      REAL spansq, cellspan, cellmean, wt_cruise_mean, wsumsq_ijl, wsumsq

      REAL*8 xax1,xaxf, yax1,yaxf, tax1,taxf
      INTEGER indx(4,2)

*  Dimension the work arrays

      REAL*8 xax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 yax(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL*8 tax(wrk3lox:wrk3hix/2, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL obs_cruise(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)
      REAL cell_avg(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .               wrk5loz:wrk5hiz, wrk5lot:wrk5hit)
      REAL wi_sums(wrk6lox:wrk6hix, wrk6loy:wrk6hiy,
     .               wrk6loz:wrk6hiz, wrk6lot:wrk6hit)
      REAL cruisespan(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .               wrk7loz:wrk7hiz, wrk7lot:wrk7hit)
      REAL cruisemean(wrk8lox:wrk8hix, wrk8loy:wrk8hiy,
     .               wrk8loz:wrk8hiz, wrk8lot:wrk8hit)
      REAL span(wrk9lox:wrk9hix, wrk9loy:wrk9hiy,
     .               wrk9loz:wrk9hiz, wrk9lot:wrk9hit)

      CHARACTER*250 errtxt

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulox(4), moduloy(4), modulot(4), regular(4)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      obs_only = 0	! Compute the variance for average of variable (e.g. fco2)

*  Check to see if output axes are modulo

      CALL ef_get_axis_info (id, 6, ax_name, ax_units, backward, 
     .                       modulox, regular)
      CALL ef_get_axis_info (id, 7, ax_name, ax_units, backward, 
     .                       moduloy, regular)
      CALL ef_get_axis_info (id, 8, ax_name, ax_units, backward, 
     .                       modulot, regular)

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
            IF (npts.NE.nxpts) GOTO 900
         ENDIF
 110  CONTINUE

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

      call ef_get_coordinates(id, ARG6, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG6), arg_hi_ss(X_AXIS, ARG6), xax)

      call ef_get_coordinates(id, ARG7, Y_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG7), arg_hi_ss(Y_AXIS, ARG7), yax)

      call ef_get_coordinates(id, ARG8, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG8), arg_hi_ss(T_AXIS, ARG8), tax)

*  Set start, end, and delta for output axes. 

      x1 = xax(1,1,1,1)
      y1 = yax(1,1,1,1)
      t1 = tax(1,1,1,1)

      xf = xax(nx,1,1,1)
      yf = yax(ny,1,1,1)
      tf = tax(nt,1,1,1)

*  Gridbox sizes in data units
      dx = xf - x1
      IF (nx .GT. 1) dx = (xf-x1)/ REAL(nx-1)
      dy = yf - y1
      IF (ny .GT. 1) dy = (yf-y1)/ REAL(ny-1) 
      dt = tf - t1
      IF (nt .GT. 1) dt = (tf-t1)/ REAL(nt-1) 

      IF (nx .EQ. 1) THEN 
         CALL ef_get_box_limits(id, ARG6, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG6), arg_hi_ss(X_AXIS, ARG6), xax1,xaxf)
         dx = xaxf - xax1
      ENDIF
      IF (ny .EQ. 1) THEN 
         CALL ef_get_box_limits(id, ARG7, Y_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG7), arg_hi_ss(Y_AXIS, ARG7), yax1,yaxf)
         dy = yaxf - yax1
      ENDIF
         IF (nt .EQ. 1) THEN 
         CALL ef_get_box_limits(id, ARG8, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG8), arg_hi_ss(T_AXIS, ARG8), tax1,taxf)
         tx = taxf - tax1
      ENDIF

*  Compute result

*  Initialize counts.

      DO L = 1, nt
      DO k = 1, 2
      DO j = 1, ny
      DO i = 1, nx
         cell_avg(i,j,k,L) = 0.
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      DO L = 1, nt
      DO k = 1, 2
      DO j = 1, ny
      DO i = 1, nx
         obs_cruise(i,j,k,L) = 0.
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      DO L = 1, nt
      DO k = 1, 1
      DO j = 1, ny
      DO i = 1, nx
         wi_sums(i,j,k,L) = 0.
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      DO L = 1, nt
      DO k = 1, wrk8hiz
      DO j = 1, ny
      DO i = 1, nx
         cruisemean(i,j,k,L) = 0.
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      DO L = 1, nt
      DO k = 1, wrk9hiz
      DO j = 1, ny
      DO i = 1, nx
         span(i,j,k,L) = 0.
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      DO L = 1, nt
      DO j = 1, ny
      DO i = 1, nx
         cruisespan(i,j,1,L) =  1000
         cruisespan(i,j,2,L) = -1000.
         cruisespan(i,j,3,L) =  1000.
         cruisespan(i,j,4,L) = -1000.
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

* We have come to the end of a cruise loop over cells represented by the last cruise,
* summing and counting the observations in the cruise, computing weights. 
* weights are span of cruise across cell
* cruisemean is weighted cruise mean in cell
* cell_avg(i,j,1,L) weighted cruise mean averaged over cruises in cell
* cell_avg(i,j,2,L) is count of cruises in cell
* wi_sums(i,j,1,L) sum of weights wi in cell

         IF (ncc .NE. icruise) THEN  
	    icend = n-1

            DO L = indx(4,1), indx(4,2)
            DO j = indx(2,1), indx(2,2)
            DO i = indx(1,1), indx(1,2)
	       IF (obs_cruise(i,j,2,L) .GT. 0.) THEN
                  spansq = (cruisespan(i,j,2,l)-cruisespan(i,j,1,l))**2 +
     .                     (cruisespan(i,j,4,l)-cruisespan(i,j,3,l))**2
                  cellspan = SQRT(spansq)

                  wt_cruise_mean = cellspan*
     .                obs_cruise(i,j,1,L)/obs_cruise(i,j,2,L)

                  cell_avg(i,j,1,L) = cell_avg(i,j,1,L) + wt_cruise_mean
                  cell_avg(i,j,2,L) = cell_avg(i,j,2,L) + 1.  

		  wi_sums(i,j,1,l) = wi_sums(i,j,1,l) + cellspan

		  nc_cell = cell_avg(i,j,2,L)
		  IF (nc_cell .GT. wrk8hiz) THEN
                     print *, 'Num_obs in cell at I,J,L = ', i,j,L, 
     .               ' is', nc_cell
		     CALL ef_bail_out (id,
     .      'too many cruises in cell. increase K size of work array 8')
		  ENDIF
		  cruisemean(i,j,nc_cell,l) = wt_cruise_mean
		  span(i,j,nc_cell,l) = cellspan

	       ENDIF

            ENDDO
            ENDDO
            ENDDO

	    icruise = ncc

            DO L = indx(4,1), indx(4,2)
            DO j = indx(2,1), indx(2,2)
            DO i = indx(1,1), indx(1,2)
	       obs_cruise(i,j,1,L) = 0.
	       obs_cruise(i,j,2,L) = 0.
            ENDDO
            ENDDO
            ENDDO

* reset indices
            indx(1,1) = nx
            indx(1,2) = 1
            indx(2,1) = ny
            indx(2,2) = 1
            indx(4,1) = nt
            indx(4,2) = 1

	 ENDIF

*  Count and add up observations in each cell, in this cruise. Span computes the arc
*  length of the current cruise.

         CALL nobsxytv_span (obs_only, xx, yy, tt, vv, nx, ny, nt, 
     .           x1, y1, t1, dx, dy, dt, xf, yf, tf, indx, obs_cruise,
     .           cruisespan)

 300     CONTINUE

* Get data from the last cruise

	icend = n-1

	DO L = indx(4,1), indx(4,2)
	DO j = indx(2,1), indx(2,2)
	DO i = indx(1,1), indx(1,2)

	       IF (obs_cruise(i,j,2,L) .GT. 0.) THEN
                  spansq = (cruisespan(i,j,2,l)-cruisespan(i,j,1,l))**2 +
     .                     (cruisespan(i,j,4,l)-cruisespan(i,j,3,l))**2
                  cellspan = SQRT(spansq)

                  wt_cruise_mean = cellspan*
     .                obs_cruise(i,j,1,L)/obs_cruise(i,j,2,L)

                  cell_avg(i,j,1,L) = cell_avg(i,j,1,L) + wt_cruise_mean
                  cell_avg(i,j,2,L) = cell_avg(i,j,2,L) + 1.  

		  wi_sums(i,j,1,l) = wi_sums(i,j,1,l) + cellspan

		  nc_cell = cell_avg(i,j,2,L)
		  IF (nc_cell .GT. wrk8hiz) THEN
                     print *, 'Num_obs in cell at I,J,L = ', i,j,L, 
     .               ' is', nc_cell
		     CALL ef_bail_out (id,
     .      'too many cruises in cell. increase K size of work array 8')
		  ENDIF
		  cruisemean(i,j,nc_cell,l) = wt_cruise_mean
		  span(i,j,nc_cell,l) = cellspan

	       ENDIF

	ENDDO
	ENDDO
	ENDDO
      
*  Put gridded fcn into result variable.
* wi_sums(i,j,1,L) sum of weights wi in cell
* V1 sum of normalized weights
* V2 sum of normalized weights squared
* for s^2  wsumsq sum of wi*(xi - weighted_mean)**2

      k = res_lo_ss(Z_AXIS)
      DO 420 L = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         DO 410 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            DO 400 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

               IF (cell_avg(i,j,2,L) .EQ. 0) THEN
                   result(i,j,k,l) = bad_flag_result
               ELSE IF (cell_avg(i,j,2,L) .EQ. 1) THEN
                   result(i,j,k,l) = 0.

* Compute sum of (cruise-cell-averages - weighted cell mean) squared\

               ELSE IF (cell_avg(i,j,2,L) .GT. 1) THEN

                  cellmean = cell_avg(i,j,1,L)/wi_sums(i,j,1,l)

                  wsumsq_ijl = 0.
                  wsumsq = 0.
		  v1 = 0.
                  DO 500, nk = 1, cell_avg(i,j,2,L)
                      cellspan = span(i,j,nk,L)
		      wt_cruise_mean = cruisemean(i,j,nk,L)
		      unwtd_mean = 0.
		      IF (cellspan .NE. 0.) unwtd_mean = wt_cruise_mean/cellspan
		      spansq = cellspan*cellspan

		      wsumsq = wsumsq + spansq

* sum the squares of difference of unweighted indiv. means minus weighted mean,

		      difsum = unwtd_mean - cellmean
                      wsumsq_ijl = wsumsq_ijl + 
     .                  cellspan*(difsum)**2

 500              CONTINUE
                  V1 = wi_sums(i,j,1,l)
                  V2 = wsumsq
                  denom = V1*V1 - V2

                  IF (denom .NE. 0.) THEN
                     result(i,j,k,l) = wsumsq_ijl* V1/denom
                  ELSE
                     result(i,j,k,l) = bad_flag_result
                  ENDIF
	       ENDIF

 400        CONTINUE

 410     CONTINUE
 420  CONTINUE

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

      SUBROUTINE pickoutxytcv (obs_only, vars, alon, alat, atimes,
     .   cruises, n, vv, xx, yy, tt, ncc)

*  Called from the scatter* family of gridding functions.
*  Pick out nth item from a set of arrays, which are really 
*  4-dimensioned arrays with normal axes on three of the four axes.

      REAL alon(*), alat(*), atimes(*), cruises(*), vars(*)
      REAL xx, yy, tt, vv
      INTEGER obs_only, n, ncc

      vv = 1.
      IF (obs_only .EQ. 0) vv = vars(n)
      xx = alon(n)
      yy = alat(n)
      tt = atimes(n)
      ncc = cruises(n)

      RETURN
      END



c***********************************************************************
c...NOBS.....Based on gausswt, return only the # of obs in (x,y) gridbox
c............method is to call this sub for each data value
c............all calcs done in gridbox units
c............xx/x1/xf all in same units
c............yy/y1/yf all in same units

c  i	xx,yy,tt=x/y/t location of data pt (data units)
c  i	nx,ny,nt=size of grids
c  i	x1,y1,L1=west/south/earliest edge of grid (center of 1st box in data units)
c  o 	obs_cruise(nx,ny,nt) number of obs in each (x,y,t) gridbox
c  o 	cellspan arc length for cruise in current xyt gridbox
c--------------------------------------------------------------------------

	SUBROUTINE nobsxytv_span (obs_only, xx, yy, tt, vv, nx, ny, nt, 
     .             x1, y1, t1, dx, dy, dt, xf, yf, tf, indx, obs_cruise,
     .             cruisespan)


* Argument declarations

        INTEGER nx, ny, nt, obs_only, indx(4,2)
	REAL xx, yy, tt, vv, x1, y1, t1, dx, dy, dt, xf, 
     .       yf, tf, obs_cruise(nx,ny,2,nt), cruisespan(nx,ny,4,nt)

* Local variable declarations

        REAL xxg, yyg, ttg
        INTEGER ig, jg, Lg

	xxg = (xx-x1)/ dx + 1.		  ! grid values of data location
	yyg = (yy-y1)/ dy + 1.
	ttg = (tt-t1)/ dt + 1.

c.................sum the value and count in each (x,y,t) gridbox
	ig = NINT(xxg)
	jg = NINT(yyg)
	Lg = NINT(ttg)
	IF (ig.LT.1 .OR. ig.GT.nx) RETURN
	IF (ig.LT.1 .OR. ig.GT.nx) RETURN
	IF (lg.LT.1 .OR. lg.GT.nt) RETURN


	IF (obs_only .EQ. 0) THEN
           IF (ig.GE.1 .AND. ig.LE.nx .AND. jg.GE.1 .AND. jg.LE.ny .AND.
     .       Lg.GE.1 .AND. Lg.LE.nt   ) THEN
     		obs_cruise(ig,jg,1,Lg) = obs_cruise(ig,jg,1,Lg) + vv
     		obs_cruise(ig,jg,2,Lg) = obs_cruise(ig,jg,2,Lg) + 1.
	   ENDIF
	ELSE
	   IF (ig.GE.1 .AND. ig.LE.nx .AND. jg.GE.1 .AND. jg.LE.ny .AND.
     .      Lg.GE.1 .AND. Lg.LE.nt   ) THEN
     		obs_cruise(ig,jg,1,Lg) = obs_cruise(ig,jg,1,Lg) + 1.
	   ENDIF
	ENDIF

* Keep the min and max lon and lat, to use in computing the span of the
* cruise in this grid cell.
	IF (ig.GE.1 .AND. ig.LE.nx .AND. jg.GE.1 .AND. jg.LE.ny .AND.
     .       Lg.GE.1 .AND. Lg.LE.nt ) THEN
           cruisespan(ig,jg,1,Lg) = MIN(cruisespan(ig,jg,1,Lg), xx)
           cruisespan(ig,jg,2,Lg) = MAX(cruisespan(ig,jg,2,Lg), xx)
           cruisespan(ig,jg,3,Lg) = MIN(cruisespan(ig,jg,3,Lg), yy)
           cruisespan(ig,jg,4,Lg) = MAX(cruisespan(ig,jg,4,Lg), yy)
	ENDIF

* Store the min and max index used in this cruise
	indx(1,1) = MIN(indx(1,1), ig)
	indx(1,2) = MAX(indx(1,2), ig)
	indx(2,1) = MIN(indx(2,1), jg)
	indx(2,2) = MAX(indx(2,2), jg)
	indx(4,1) = MIN(indx(4,1), Lg)
	indx(4,2) = MAX(indx(4,2), Lg)

	RETURN
	END
