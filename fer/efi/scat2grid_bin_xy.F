*
*  scat2grid_bin_xy.F
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
* February 1, 2008 
* Ferret V665 10/2010 Change loop structure to make this more efficient.

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


      SUBROUTINE  scat2grid_bin_xy_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Put scattered data into XY grid by binning')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 5)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .      NORMAL, NORMAL)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 5)

* Output grid is determined by arguments 4 and 5, the result's x and y axes. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'X coordinates of scattered XY locations')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Y coordinates of scattered XY locations')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      WRITE (buff, 20) 
   20 FORMAT ('F(X,Y) Data at scattered input XY locations')
      CALL ef_set_arg_desc(id, arg, buff)
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'X axis coordinates of output grid')
      CALL ef_set_axis_influence(id, arg, YES, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'YAXPTS')
      CALL ef_set_arg_desc(id, arg, 
     .                   'Y axis coordinates of output grid')
      CALL ef_set_axis_influence(id, arg, NO, YES, NO, NO)



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
      SUBROUTINE scat2grid_bin_xy_work_size(id)

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
      INTEGER nxout, nyout, nx2, ny2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4)
      nyout = 1 + arg_hi_ss(Y_AXIS,ARG5) - arg_lo_ss(Y_AXIS,ARG5)

      nx2 = nxout* 2
      ny2 = nyout* 2


* xaxlo  output x axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nx2, 1, 1, 1)
* xaxhi  output x axis
      CALL ef_set_work_array_dims (id, 2, 1, 1, 1, 1, nx2, 1, 1, 1)

* yaxlo  output y axis
      CALL ef_set_work_array_dims (id, 3, 1, 1, 1, 1, ny2, 1, 1, 1)
* yaxhi  output y axis
      CALL ef_set_work_array_dims (id, 4, 1, 1, 1, 1, ny2, 1, 1, 1)

* counts
      CALL ef_set_work_array_dims (id, 5, 1, 1, 1, 1, nxout, nyout, 1, 1) 

      RETURN
      END

* In this subroutine we compute the result
*

      SUBROUTINE  scat2grid_bin_xy_compute(id, arg_1, arg_2, arg_3, 
     .     arg_4, arg_5, result, xaxlo, xaxhi, yaxlo, yaxhi, counts)

*  arg_1  xpts  \
*  arg_2  ypts  / Scattered x,y triples to be gridded.
*  arg_3  fpts /
*  arg_4  xaxis of new grid   
*  arg_5  yaxis of new grid

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

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
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz, 
     .     mem5lot:mem5hit)

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

*  Dimension the work arrays

      REAL*8 xaxlo(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 xaxhi(wrk2lox:wrk2hix/2, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL*8 yaxlo(wrk3lox:wrk3hix/2, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL*8 yaxhi(wrk4lox:wrk4hix/2, wrk4loy:wrk4hiy, 
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)
      REAL counts(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .              wrk5loz:wrk5hiz, wrk5lot:wrk5hit) 

      INTEGER i, j, k, l, m, n
      INTEGER i1, i2, j1, k1, l1
      INTEGER i4, i4n, j5, j5n

      INTEGER nxpts, nypts, nscat, nput
      INTEGER nx, ny
      INTEGER i1n, i2n

      REAL xx, yy, ff

      CHARACTER*250 errtxt

*  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(5)*16, ax_units(5)*16
      LOGICAL backward(5), modulox(5), moduloy(5), regular(5)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


*  Check to see if output axes are modulo

      CALL ef_get_axis_info (id, 4, ax_name, ax_units, backward, 
     .                       modulox, regular)

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, Z, or T axis of the input arguments.

      nxpts = 0
      nypts = 0

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
            if (i2n-i2 .NE. 0) nypts = 1 + (i2n - i2)
         ENDIF
 110  CONTINUE

      IF (nxpts .NE. nypts .OR. nxpts .EQ. 0) GOTO 900

      nscat = nxpts

*  Compute number of points in output axes.

      i4 = arg_lo_ss(X_AXIS,ARG4)
      i4n = arg_hi_ss(X_AXIS,ARG4)

      j5 = arg_lo_ss(Y_AXIS,ARG5)
      j5n = arg_hi_ss(Y_AXIS,ARG5)

      nx = 1 + (i4n - i4)
      ny = 1 + (j5n - j5)

*  Check that xax is a X axis and yax a Y axis

      IF (i4 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Fourth argument must be a X axis'
         GO TO 999
      ENDIF
      IF (j5 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Fifth argument must be a Y axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axes.

      CALL ef_get_box_limits(id, ARG4, X_AXIS,
     .   arg_lo_ss(X_AXIS, ARG4), arg_hi_ss(X_AXIS, ARG4), xaxlo, xaxhi)

      CALL ef_get_box_limits(id, ARG5, Y_AXIS,
     .   arg_lo_ss(Y_AXIS, ARG5), arg_hi_ss(Y_AXIS, ARG5), yaxlo, yaxhi)

*  Compute result

      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)

*  Initialize result.

      DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
         DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
            result(i,j,k,l) = 0.0
         ENDDO
      ENDDO

*  Initialize counts.

      DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
         DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
            counts(i,j,1,1) = 0.0
         ENDDO
      ENDDO

*  Loop over x and y, placing the function values in the appropriate slots
*  in the grid. xpts and ypts may be on the X,Y,Z or T axis of ARG1 
*  and ARG2, sending them to a subroutine collapses the extra dimensions so the
*  value can be found.

      nput = 0
      DO 300 n = 1, nscat

         CALL pickout3 (arg_1, arg_2, arg_3, n, xx, yy, ff)

*  If output X axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.

         IF (modulox(1) .AND. xx.NE.bad_flag(ARG1)) 
     .      CALL modscat (xaxlo, nx, 1, xx)

*  Put fcn into result variable.

         IF (ff .NE. bad_flag(ARG3)) THEN
            i1 = 1
            DO 510 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
               IF( xx.GE.xaxlo(i1,1,1,1) .AND.
     .             xx.LT.xaxhi(i1,1,1,1) ) THEN

                  j1 = 1
                  DO 500 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
                     IF (yy.GE.yaxlo(j1,1,1,1) .AND.
     .                   yy.LT.yaxhi(j1,1,1,1) )  THEN
                        result(i,j,k,l) = result(i,j,k,l) + ff
                        counts(i,j,1,1) = counts(i,j,1,1) + 1. 
                     ENDIF

                     j1 = j1 + 1

 500              CONTINUE

	       ENDIF ! if xx

               i1 = i1 + 1
 510        CONTINUE
         ENDIF
300   CONTINUE

* Compute result, average in each bin.

      i1 = 1
      DO 610 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         j1 = 1
         DO 600 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

               IF( counts(i,j,1,1) .GT. 0. ) THEN
                  result(i,j,k,l) = result(i,j,k,l)/counts(i,j,1,1)
               ELSE
                  result(i,j,k,l) = bad_flag_result
               ENDIF

            j1 = j1 + 1
600      CONTINUE
         i1 = i1 + 1
610   CONTINUE

      RETURN

900   CONTINUE

      IF (nxpts .NE. nypts) THEN
         WRITE (errtxt,20) nxpts, nypts
      ELSE IF (nxpts .EQ. 0) THEN
         WRITE (errtxt, 30)
      ENDIF
 20   FORMAT ('Input scattered x, y have different # of points', 2I8)
 30   FORMAT ('No data in scattered x, y points')
      GOTO 999 

 999  CALL EF_BAIL_OUT(id, errtxt)

      RETURN
      END
