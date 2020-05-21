*
*  tracks2grid_std_xyt.F
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
* 9/ 3/2015 Fix LAS/SOCAT ticket 1720, missingness patterns in cruise-weighted 
*           vs unweighted gridded fields.
* 1/14/2016 Fix LAS/SOCAT ticket 1721, missing/zero where there should be data in cell
* 1/27/2016 Apply modulo operation to scattered longitudes or time if the output
*           axis is a modulo axis.
* 
* DSGTODO  Make use of dsg-handling upgrade
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
* This function will return the std for the averages, based on weighted means,
* for small samples, http://wapedia.mobi/en/Weighted_mean#6. and
* http://wapedia.mobi/en/Weighted_mean#7.
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


      SUBROUTINE  tracks2grid_std_xyt_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Weighted std by cruise, when gridding scattered ',
     .        'data to XYT grid')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 8)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS,
     .        IMPLIED_BY_ARGS, NORMAL, IMPLIED_BY_ARGS)

      CALL ef_set_num_work_arrays(id, 8)

* Output grid is determined by arguments 4, 5, 6, the result's x, y, t axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'VARIABLE')
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
      SUBROUTINE tracks2grid_std_xyt_work_size(id)

      IMPLICIT NONE
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

* xax  output x axis cell edges
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nxout, 2, 1, 1)

* yax  output y axis cell edges
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, nyout, 2, 1, 1)

* Tax  output t axis cell edges
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, ntout, 2, 1, 1)

* obs_cruise  work array - sum and counts of observations per cruise
* x,y,t direction: output grid
* z direction: 1..3
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, 
     .                             nxout, nyout, 3, ntout)

* cell_avg  work array - work array - sum and counts over cruises for cell average
* x,y,t direction: output grid
* z direction: 1..2
      CALL ef_set_work_array_dims (id, 5, 1, 1, 1, 1, 
     .                             nxout, nyout, 2, ntout)

* wi_sums k=1, sum of cruiseweight wi = sum of obs in the cells
* x,y,t direction: output grid
* z direction: 1
      CALL ef_set_work_array_dims (id, 6, 1, 1, 1, 1, 
     .                             nxout, nyout, 1, ntout)

* cruisemean saving individual weighted cruise means. 
* for computing s^2  sum of wi*(xi - weighted_mean)**2
* x,y,t direction: output grid
* z direction: 30 (max cruises per cell?)
      CALL ef_set_work_array_dims (id, 7, 1, 1, 1, 1, 
     .                             nxout, nyout, 30, ntout)

* cruiseweight saving individual cruiseweight. 
* for computing s^2  sum of wi*(xi - weighted_mean)**2
* x,y,t direction: output grid
* z direction: 30 (max cruises per cell?)
      CALL ef_set_work_array_dims (id, 8, 1, 1, 1, 1, 
     .                             nxout, nyout, 30, ntout)

      RETURN
      END

*
* In this subroutine we compute the result
*

      SUBROUTINE tracks2grid_std_xyt_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, arg_5, arg_6, arg_7, arg_8, result, xax, yax, tax, 
     .     obs_cruise, cell_avg, wi_sums, cruisemean, cruiseweight)

*  arg_1  var
*  arg_2  xpts  \
*  arg_3  ypts   | Scattered x,y triples to be gridded.
*  arg_4  tpts  / 
*  arg_5  icruise
*
*  arg_6  xaxis of new grid   
*  arg_7  yaxis of new grid
*  arg_8  taxis of new grid

      IMPLICIT NONE
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
      INTEGER i6,j7,L8,i6n,j7n,L8n

      INTEGER nxpts, npts
      INTEGER nx, ny, nt
      INTEGER i1, i2, i1n, i2n
      INTEGER obs_only, icruise, ncc, nc_cell, nk

      REAL x1, y1, t1, xf, yf, tf
      REAL xx, yy, tt, vv

*  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), mdulo(6), regular(6)
      LOGICAL modulox, modulot
      REAL*8 modxlen, modtlen

      REAL dx, dy, xxbeg, xxend, yybeg, yyend
      REAL cellmean, wt_cruise_mean, unwtd_mean,
     .     wsumsq_ijl, wsumsq, diffsum, wtcruise,
     .     denom, v1, v2, wtsq

      REAL*8 xax1,xaxf, yax1,yaxf, tax1,taxf
      INTEGER indx(4,2), icstart, icend, nscat

*  Dimension the work arrays

      REAL*8 xax(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 yax(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL*8 tax(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL obs_cruise(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)
      REAL cell_avg(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .               wrk5loz:wrk5hiz, wrk5lot:wrk5hit)
      REAL wi_sums(wrk6lox:wrk6hix, wrk6loy:wrk6hiy,
     .               wrk6loz:wrk6hiz, wrk6lot:wrk6hit)
      REAL cruisemean(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .               wrk7loz:wrk7hiz, wrk7lot:wrk7hit)
      REAL cruiseweight(wrk8lox:wrk8hix, wrk8loy:wrk8hiy,
     .               wrk8loz:wrk8hiz, wrk8lot:wrk8hit)

      CHARACTER*250 errtxt

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

      obs_only = 0	! Compute the std for average of variable (e.g. fco2)

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

      CALL EF_GET_BOX_LIMITS(id, ARG6, X_AXIS,
     .   arg_lo_ss(X_AXIS, ARG6), arg_hi_ss(X_AXIS, ARG6), xax(1,1,1,1), xax(1,2,1,1))

      CALL EF_GET_BOX_LIMITS(id, ARG7, Y_AXIS,
     .   arg_lo_ss(Y_AXIS, ARG7), arg_hi_ss(Y_AXIS, ARG7), yax(1,1,1,1), yax(1,2,1,1))

      CALL EF_GET_BOX_LIMITS(id, ARG8, T_AXIS,
     .   arg_lo_ss(T_AXIS, ARG8), arg_hi_ss(T_AXIS, ARG8), tax(1,1,1,1), tax(1,2,1,1))

*  Get first coordinates of output axes.

      call ef_get_coordinates(id, ARG6, X_AXIS, 
     .   arg_lo_ss(X_AXIS, ARG6), arg_lo_ss(X_AXIS, ARG6), xax1)
      call ef_get_coordinates(id, ARG7, X_AXIS, 
     .   arg_lo_ss(Y_AXIS, ARG7), arg_lo_ss(Y_AXIS, ARG7), yax1)
      x1 = xax1
      y1 = yax1

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
      DO k = 1, 3
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
      DO k = 1, wrk7hiz
      DO j = 1, ny
      DO i = 1, nx
         cruisemean(i,j,k,L) = 0.
      ENDDO
      ENDDO
      ENDDO
      ENDDO

      DO L = 1, nt
      DO k = 1, wrk7hiz
      DO j = 1, ny
      DO i = 1, nx
         cruiseweight(i,j,k,L) = 0.
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

*  If output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
      IF ( modulox ) THEN
         CALL modscat(xax(1,1,1,1), xax(nx,2,1,1), modxlen, 1, xx)
      ENDIF
      IF ( modulot ) THEN
         CALL modscat(tax(1,1,1,1), tax(nt,2,1,1), modtlen, 1, tt)
      ENDIF

      icruise = ncc

* For each cruise keep track of the index range it spans in the result grid
* Then at the end of the cruise need only to process that subset of the grid.
      indx(1,1) = nx
      indx(1,2) = 1
      indx(2,1) = ny
      indx(2,2) = 1
      indx(4,1) = nt
      indx(4,2) = 1

      icstart = 1
      DO 300 n = 1, nscat

         CALL pickoutxytcv (obs_only, arg_1, arg_2, arg_3, arg_4, arg_5,
     .           n, vv, xx, yy, tt, ncc)
         IF (vv .EQ. bad_flag(ARG1) .OR. xx .EQ. bad_flag(ARG2) .OR. 
     .          yy .EQ. bad_flag(ARG3) .OR. tt .EQ. bad_flag(ARG4) ) GOTO 300

*  If output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
         IF ( modulox ) THEN
            CALL modscat(xax(1,1,1,1), xax(nx,2,1,1), modxlen, 1, xx)
         ENDIF
         IF ( modulot ) THEN
            CALL modscat(tax(1,1,1,1), tax(nt,2,1,1), modtlen, 1, tt)
         ENDIF

* We have come to the end of a cruise loop over cells represented by the last cruise,
* summing and counting the observations in the cruise, computing cruiseweight. 
* cruisemean is weighted cruise mean in cell
* cruiseweight are variances for cruise mean in cell
* cell_avg(i,j,1,L) cruise mean averaged over cruises in cell
* cell_avg(i,j,2,L) is count of cruises in cell
* wi_sums(i,j,1,L) sum of cruiseweight wi in cell

         IF (ncc .NE. icruise) THEN  
	    icend = n-1

            DO L = indx(4,1), indx(4,2)
            DO j = indx(2,1), indx(2,2)
            DO i = indx(1,1), indx(1,2)

	       IF (obs_cruise(i,j,2,L).GT.0. .AND. obs_cruise(i,j,3,L).NE.0.) THEN

                  wt_cruise_mean = obs_cruise(i,j,1,L)/obs_cruise(i,j,3,L)

                  unwtd_mean = obs_cruise(i,j,1,L) ! nobsxytv_var returns mean already computed

                  cell_avg(i,j,1,L) = cell_avg(i,j,1,L) + wt_cruise_mean
                  cell_avg(i,j,2,L) = cell_avg(i,j,2,L) + 1.  

		  wi_sums(i,j,1,l) = wi_sums(i,j,1,l) + 1./obs_cruise(i,j,3,L)

		  nc_cell = cell_avg(i,j,2,L)
		  IF (nc_cell.GT.wrk7hiz) THEN
                     print *, 'Num_obs in cell at I,J,L = ', i,j,L, 
     .               ' is', nc_cell
		     CALL ef_bail_out (id,
     .      'too many cruises in cell. increase K size of work array 7,8')
		  ENDIF

		  cruisemean(i,j,nc_cell,l) = unwtd_mean
		  cruiseweight(i,j,nc_cell,l) = 1./obs_cruise(i,j,3,L)

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
	       obs_cruise(i,j,3,L) = 0.
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

*  Count and add up observations in each cell, in this cruise. 
     
         CALL nobsxytv_var (obs_only, xx, yy, tt, vv, nx, ny, nt, 
     .        xax(1,1,1,1), xax(1,2,1,1), yax(1,1,1,1), yax(1,2,1,1), 
     .        tax(1,1,1,1), tax(1,2,1,1), indx, obs_cruise)

 300  CONTINUE

* Get data from the last cruise

      icend = n-1

      DO L = indx(4,1), indx(4,2)
         DO j = indx(2,1), indx(2,2)
            DO i = indx(1,1), indx(1,2)

	       IF (obs_cruise(i,j,2,L).GT.0. .AND. obs_cruise(i,j,3,L).NE.0.) THEN

                  wt_cruise_mean = obs_cruise(i,j,1,L)/obs_cruise(i,j,3,L)

                  unwtd_mean = obs_cruise(i,j,1,L) ! nobsxytv_var returns mean already computed

                  cell_avg(i,j,1,L) = cell_avg(i,j,1,L) + wt_cruise_mean
                  cell_avg(i,j,2,L) = cell_avg(i,j,2,L) + 1.  

		  wi_sums(i,j,1,l) = wi_sums(i,j,1,l) + 1./obs_cruise(i,j,3,L)

		  nc_cell = cell_avg(i,j,2,L)
		  IF (nc_cell.GT.wrk7hiz) THEN
                     print *, 'Num_obs in cell at I,J,L = ', i,j,L, 
     .               ' is', nc_cell
		     CALL ef_bail_out (id,
     .      'too many cruises in cell. increase K size of work array 7,8')
		  ENDIF

		  cruisemean(i,j,nc_cell,l) = unwtd_mean
		  cruiseweight(i,j,nc_cell,l) = 1./obs_cruise(i,j,3,L)

	       ENDIF

            ENDDO
         ENDDO
      ENDDO
      
*  Put gridded fcn into result variable.
* wi_sums(i,j,1,L) sum of cruiseweight wi in cell
* V1 sum of cruiseweight
* V2 sum of cruiseweight squared
* for s^2  wsumsq sum of wi*(xi - weighted_mean)**2

      k = res_lo_ss(Z_AXIS)
      DO 420 L = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         DO 410 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            DO 400 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

               nc_cell = cell_avg(i,j,2,L)

               IF (nc_cell .EQ. 0) THEN
                   result(i,j,k,l) = bad_flag_result
               ELSE IF (nc_cell .EQ. 1) THEN
                   result(i,j,k,l) = 0.
c               ELSE IF (wi_sums(i,j,1,l).LE.1.e-6) THEN
c                   result(i,j,k,l) = bad_flag_result

* Compute sum of (cruise-cell-averages - weighted cell mean) squared\

               ELSE IF (nc_cell.GT.1 .AND. wi_sums(i,j,1,l).NE.0.) THEN

                  cellmean = cell_avg(i,j,1,L)/wi_sums(i,j,1,l)

                  wsumsq_ijl = 0.
                  wsumsq = 0.
		  v1 = 0.
                  DO 500, nk = 1, nc_cell
                      wtcruise = cruiseweight(i,j,nk,L)
		      unwtd_mean = cruisemean(i,j,nk,L)

		      wtsq = wtcruise*wtcruise
		      wsumsq = wsumsq + wtsq  

* Sum the squares of difference of unweighted cruise means 
* minus weighted cell mean

		      diffsum = unwtd_mean - cellmean
                      wsumsq_ijl = wsumsq_ijl + wtcruise*(diffsum)**2

 500              CONTINUE
                  V1 = wi_sums(i,j,1,l)
                  V2 = wsumsq
                  denom = V1*V1 - V2
		  
c                  IF (ABS(denom).GT.1.e-6) THEN
                  IF (ABS(denom).NE.0.) THEN
                     result(i,j,k,l) = SQRT(wsumsq_ijl* V1/denom)
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
