*
*  scat2grid_bin_xy
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

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .        'Put scattered data into XY grid by binning')
      CALL ef_set_num_args(id, 5)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                NORMAL,          NORMAL,
     .                                NORMAL,          NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 5)

* Output grid is determined by arguments 4 and 5, the result's x and y axes.

      arg = 1
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'X coordinates of scattered XY locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Y coordinates of scattered XY locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'F')
      CALL ef_set_arg_desc(id, arg,
     .        'variable at scattered XY locations')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'XAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'X axis coordinates of the output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, NO, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'YAXPTS')
      CALL ef_set_arg_desc(id, arg,
     .        'Y axis coordinates of the output grid')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, NO, NO, NO, NO)

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

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays, X/Y/Z/T/E/F dimensions
*
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER nxout, nyout, nx2, ny2

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nxout = 1 + arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4)
      nyout = 1 + arg_hi_ss(Y_AXIS,ARG5) - arg_lo_ss(Y_AXIS,ARG5)

      nx2 = nxout * 2
      ny2 = nyout * 2

* xaxlo  output x axis
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                              nx2, 1, 1, 1, 1, 1)

* xaxhi  output x axis
      CALL ef_set_work_array_dims_6d(id, 2,
     .                                1, 1, 1, 1, 1, 1,
     .                              nx2, 1, 1, 1, 1, 1)

* yaxlo  output y axis
      CALL ef_set_work_array_dims_6d(id, 3,
     .                                1, 1, 1, 1, 1, 1,
     .                              ny2, 1, 1, 1, 1, 1)

* yaxhi  output y axis
      CALL ef_set_work_array_dims_6d(id, 4,
     .                                1, 1, 1, 1, 1, 1,
     .                              ny2, 1, 1, 1, 1, 1)

* counts
      CALL ef_set_work_array_dims_6d(id, 5,
     .                                1,     1, 1, 1, 1, 1,
     .                            nxout, nyout, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE scat2grid_bin_xy_compute(id, arg_1, arg_2, arg_3,
     .                          arg_4, arg_5, result, xaxlo, xaxhi,
     .                          yaxlo, yaxhi, counts)

*  arg_1  xpts  \
*  arg_2  ypts  / Scattered x,y triples to be gridded.
*  arg_3  variable, /
*  arg_4  xaxis of new grid
*  arg_5  yaxis of new grid

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz,
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz,
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz,
     .           mem3lot:mem3hit, mem3loe:mem3hie, mem3lof:mem3hif)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz,
     .           mem4lot:mem4hit, mem4loe:mem4hie, mem4lof:mem4hif)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz,
     .           mem5lot:mem5hit, mem5loe:mem5hie, mem5lof:mem5hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* Ignore the unused final dimensions in the work arrays since
* Fortran is column major
      REAL*8 xaxlo(wrk1lox:wrk1hix/2)
      REAL*8 xaxhi(wrk2lox:wrk2hix/2)
      REAL*8 yaxlo(wrk3lox:wrk3hix/2)
      REAL*8 yaxhi(wrk4lox:wrk4hix/2)
      REAL counts(wrk5lox:wrk5hix, wrk5loy:wrk5hiy)

* After initialization, the 'res_' arrays contain indexing information
* for the result axes.  The 'arg_' arrays will contain the indexing
* information for each variable''s axes.

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1
      INTEGER lo_ss, hi_ss
      INTEGER nxpts, nypts, nscat
      INTEGER idx, nx, ny
      REAL xx, yy, ff
      CHARACTER*250 errtxt

*  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), mdulo(6), regular(6)
      LOGICAL modulox, moduloy
      REAL*8 modxlen, modylen

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Check to see if the output X axis is modulo (does not need to be regular)
      CALL ef_get_axis_info_6d(id, ARG4, ax_name, ax_units,
     .                         backward, mdulo, regular)
      modulox = mdulo(X_AXIS)
      IF ( modulox ) THEN
         CALL ef_get_axis_modulo_len(id, ARG4, X_AXIS, modxlen)
      ENDIF

*  Check to see if the output Y axis is modulo (does not need to be regular)
      CALL ef_get_axis_info_6d(id, ARG5, ax_name, ax_units,
     .                         backward, mdulo, regular)
      moduloy = mdulo(Y_AXIS)
      IF ( moduloy ) THEN
         CALL ef_get_axis_modulo_len(id, ARG5, Y_AXIS, modylen)
      ENDIF

*  Find number of points in scattered input points.  1-D arrays defining the
*  scattered data points may lie on the X, Y, Z, T, E, or F axis of the input
*  arguments.
*  Using as flattened arrays, so consider all axes.

      nxpts = 0
      DO 10 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG1)
         hi_ss = arg_hi_ss(idx,ARG1)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( nxpts .NE. 0 ) THEN
               nxpts = nxpts * (hi_ss - lo_ss + 1)
            ELSE
               nxpts = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  10  CONTINUE

      nypts = 0
      DO 20 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG2)
         hi_ss = arg_hi_ss(idx,ARG2)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( nypts .NE. 0 ) THEN
               nypts = nypts * (hi_ss - lo_ss + 1)
            ELSE
               nypts = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  20  CONTINUE

      nscat = 0
      DO 80 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG3)
         hi_ss = arg_hi_ss(idx,ARG3)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( nscat .NE. 0 ) THEN
               nscat = nscat * (hi_ss - lo_ss + 1)
            ELSE
               nscat = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  80  CONTINUE

      IF ( nxpts .NE. nypts ) THEN
         WRITE (errtxt, 90) ' XPTS,', ' YPTS ', nxpts, nypts
         GOTO 999
      ENDIF
      IF ( nxpts .NE. nscat ) THEN
         WRITE (errtxt, 90) ' XPTS,', ' F()  ', nxpts, nscat
         GOTO 999
      ENDIF
  90  FORMAT('Input scattered', 2A6, 'have different # of points', 2I8)

      IF ( nscat .LE. 0 ) THEN
         errtxt = 'No data in scattered x, y points'
         GOTO 999
      ENDIF

*  Compute number of points in output axes.

      lo_ss = arg_lo_ss(X_AXIS,ARG4)
      hi_ss = arg_hi_ss(X_AXIS,ARG4)
*  Check that xax is an X axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fourth argument must be an X axis'
         GOTO 999
      ENDIF
      nx = hi_ss - lo_ss + 1

      lo_ss = arg_lo_ss(Y_AXIS,ARG5)
      hi_ss = arg_hi_ss(Y_AXIS,ARG5)
*  Check that yax a Y axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Fifth argument must be a Y axis'
         GOTO 999
      ENDIF
      ny = hi_ss - lo_ss + 1

C  Get coordinates of output axes.

      CALL ef_get_box_limits(id, ARG4, X_AXIS, arg_lo_ss(X_AXIS, ARG4),
     .                       arg_hi_ss(X_AXIS, ARG4), xaxlo, xaxhi)

      CALL ef_get_box_limits(id, ARG5, Y_AXIS, arg_lo_ss(Y_AXIS, ARG5),
     .                       arg_hi_ss(Y_AXIS, ARG5), yaxlo, yaxhi)

*  Compute result

      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)
      m = res_lo_ss(E_AXIS)
      n = res_lo_ss(F_AXIS)

*  Initialize result.

      DO j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
         DO i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
            result(i,j,k,l,m,n) = 0.0
         ENDDO
      ENDDO

*  Initialize counts.

      DO j1 = 1, ny
         DO i1 = 1, nx
            counts(i1,j1) = 0.0
         ENDDO
      ENDDO

*  Loop over x and y, placing the function values in the appropriate slots
*  in the grid. xpts and ypts may be on the X,Y,Z,T,E or F axis of ARG1
*  and ARG2, sending them to a subroutine collapses the extra dimensions
*  so the value can be found.

      DO 700 idx = 1, nscat

         CALL pickout3(arg_1, arg_2, arg_3, idx, xx, yy, ff)
         IF ( xx .EQ. bad_flag(ARG1) .OR.
     .        yy .EQ. bad_flag(ARG2) .OR.
     .        ff .EQ. bad_flag(ARG3) ) GOTO 700

*  If output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
         IF ( modulox ) THEN
            CALL modscat(xaxlo(1), xaxhi(nx), modxlen, 1, xx)
         ENDIF
         IF ( moduloy ) THEN
            CALL modscat(yaxlo(1), yaxhi(ny), modylen, 1, yy)
         ENDIF

*  Put fcn into result variable.

         j1 = 1
         DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            IF ( yy .GE. yaxlo(j1) .AND. yy .LE. yaxhi(j1) ) THEN

         i1 = 1
         DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
            IF ( xx .GE. xaxlo(i1) .AND. xx .LE. xaxhi(i1) ) THEN

               result(i,j,k,l,m,n) = result(i,j,k,l,m,n) + ff
               counts(i1,j1) = counts(i1,j1) + 1.0
*              Location found and point assigned; go to the next scatter point
               GOTO 700

            ENDIF
            i1 = i1 + 1
 100     CONTINUE

            ENDIF
            j1 = j1 + 1
 200     CONTINUE

 700  CONTINUE

* Compute result, average in each bin.

      j1 = 1
      DO 210 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

      i1 = 1
      DO 110 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         IF ( counts(i1,j1) .GT. 0.0 ) THEN
            result(i,j,k,l,m,n) = result(i,j,k,l,m,n) / counts(i1,j1)
         ELSE
            result(i,j,k,l,m,n) = bad_flag_result
         ENDIF

         i1 = i1 + 1
 110  CONTINUE

         j1 = j1 + 1
 210  CONTINUE

      RETURN

 999  CALL EF_BAIL_OUT(id, errtxt)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN
      END
