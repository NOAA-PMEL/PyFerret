*
* samplexyt.F
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without anx fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and anx derivative works thereof, and
*  its supporting documentation for anx purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in anx publications that result from
*  the use of this software or in anx product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in anx advertising or publicity to endorse or promote anx products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with anx support, consulting,
*  training or assistance of anx kind with regard to the use, operation
*  and performance of this software nor to provide the user with anx
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND Anx EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR Anx SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR Anx DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
* Ansley Manke
*  -------------------------------------------------------------------
*
* This function samples 4-d data at the x, y, t locations indicated by 
* args 2, 3,4
* 1/18/08  Ansley Manke 
*         Result is abstract on the x axis, normal on the y and t axes,
*         and keeps the z axes of the input 4-d data.
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


      SUBROUTINE samplexyt_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled at a set of (X,Y,T) points, ',
     .   'using linear interpolation')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 4)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .     NORMAL, IMPLIED_BY_ARGS, NORMAL)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 3)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'variable (x,y,z,t) to sample')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 'X values of sample points')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 'Y values of sample points')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg, 'T values of sample points')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we provide information about the lo and hi
* limits associated with each abstract or custom axis.   The user 
* configurable information consists of the following:
*
* loss               lo subscript for an axis
*
* hiss               hi subscript for an axis
*

      SUBROUTINE samplexyt_result_limits(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo_l, my_hi_l
      INTEGER nx, ny, nz, nt

*     Use utility functions to get context information about the 
*     1st argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG2) - arg_lo_ss(X_AXIS, ARG2) + 1
      ny = arg_hi_ss(Y_AXIS, ARG2) - arg_lo_ss(Y_AXIS, ARG2) + 1
      nz = arg_hi_ss(Z_AXIS, ARG2) - arg_lo_ss(Z_AXIS, ARG2) + 1
      nt = arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) + 1

      my_lo_l = 1
      my_hi_l = max(nx,ny,nz,nt)

      CALL ef_set_axis_limits(id, X_AXIS, my_lo_l, my_hi_l)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE samplexyt_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_lens(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER mxh, myh, mth, mxl, myl, mtl

      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

* Allocate double the length of the axes for REAL*8 work arrays.
* For x and t axes allow yet one more point for handling modulo axes

      mxl = arg_lo_ss(X_AXIS,ARG1)
      myl = arg_lo_ss(Y_AXIS,ARG1)
      mtl = arg_lo_ss(T_AXIS,ARG1)

      mxh = mxl + 
     .     2* (arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 2)
      myh = myl + 
     .     2* (arg_hi_ss(Y_AXIS,ARG1)  - arg_lo_ss(Y_AXIS,ARG1) + 1)
      mth = mtl + 
     .     2* (arg_hi_ss(T_AXIS,ARG1)  - arg_lo_ss(T_AXIS,ARG1) + 2)

*  xaxdat
      CALL ef_set_work_array_dims (id, 1, mxl, 1, 1, 1, mxh, 1, 1, 1)

*  yaxdat
      CALL ef_set_work_array_dims (id, 2, myl, 1, 1, 1, myh, 1, 1, 1)

*  taxdat
      CALL ef_set_work_array_dims (id, 3, mtl, 1, 1, 1, mth, 1, 1, 1)


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*



      SUBROUTINE samplexyt_compute(id, arg_1, arg_2, arg_3, arg_4, 
     .      result, xaxdat, yaxdat, taxdat)

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
     .      memresloz:memreshiz, memreslot:memreshit)


* After initialization, the 'res_' arrays contain indexing information
* for the result axes.  The 'arg_' arrays will contain the indexing
* information for each variable's axes.

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      COMMON /STOR/ mxdat, mydat
      INTEGER mxdat, mydat
      INTEGER nx, nxx, nxy, nxz, nxt
      INTEGER ny, nyx, nyy, nyz, nyt
      INTEGER nt, ntx, nty, ntz, ntt
      INTEGER ndimx, ndimy, ndimt

*  Set up work arrays

      REAL*8 xaxdat(wrk1lox:wrk1lox+(wrk1hix-wrk1lox)/2,wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL*8 yaxdat(wrk2lox:wrk2lox+(wrk2hix-wrk2lox)/2,wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL*8 taxdat(wrk3lox:wrk3lox+(wrk3hix-wrk3lox)/2,wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)

      INTEGER i, j, k, l
      INTEGER i1,j1,k1,l1
      INTEGER i2,j2,k2,l2
      INTEGER i3,j3,k3,l3
      INTEGER i4,j4,k4,l4

      LOGICAL ok
      INTEGER STR_UPCASE, ic, imatch, jmatch, lmatch
      INTEGER ibot, itop, jbot, jtop, lbot, ltop
      REAL fxbot, fxtop, fbb, ftb, fbt, ftt
      REAL xbot, xtop, ybot, ytop, tbot, ttop
      REAL frac, tfrac_lo, tfrac_hi, aa, bb
      CHARACTER err_msg*255, test_str*16

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulo(4), regular(4)
      INTEGER xlo, xhi, ylo, yhi, tlo, thi
      REAL dx, dt, delmodx, delmodt, xpt, ypt, tpt

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      nxx = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      nxy = arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1
      nxz = arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2) + 1
      nxt = arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1
      nx = max(nxx, nxy, nxz, nxt)

      nyx = arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3) + 1
      nyy = arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3) + 1
      nyz = arg_hi_ss(Z_AXIS,ARG3) - arg_lo_ss(Z_AXIS,ARG3) + 1
      nyt = arg_hi_ss(T_AXIS,ARG3) - arg_lo_ss(T_AXIS,ARG3) + 1
      ny = max(nyx, nyy, nyz, nyt)

      ntx = arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4) + 1
      nty = arg_hi_ss(Y_AXIS,ARG4) - arg_lo_ss(Y_AXIS,ARG4) + 1
      ntz = arg_hi_ss(Z_AXIS,ARG4) - arg_lo_ss(Z_AXIS,ARG4) + 1
      ntt = arg_hi_ss(T_AXIS,ARG4) - arg_lo_ss(T_AXIS,ARG4) + 1
      nt = max(ntx, nty, ntz, ntt)

      ndimx = 0
      ndimy = 0
      ndimt = 0
      DO 110 i = X_AXIS,T_AXIS
         IF (arg_hi_ss(i,ARG2) - arg_lo_ss(i,ARG2) .GT.0) 
     .       ndimx = ndimx + 1
         IF (arg_hi_ss(i,ARG3) - arg_lo_ss(i,ARG3) .GT.0) 
     .       ndimy = ndimy + 1
         IF (arg_hi_ss(i,ARG4) - arg_lo_ss(i,ARG4) .GT.0) 
     .       ndimt = ndimt + 1
  110 CONTINUE

      IF (nx.NE.ny  .OR.  nx.NE.nt  .OR.
     .    ndimx.GT.1  .OR. ndimy.GT.1  .OR.  ndimt.GT.1) THEN
         WRITE (err_msg, 10)
         GO TO 999
      ENDIF

   10 FORMAT(
     . 'Arguments 2, 3, and 4 must be 1-dimensional lists of equal length')

*  Get x, y, t coordinates of the data to be sampled.

      CALL ef_get_coordinates(id, ARG1, X_AXIS,
     .   arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), xaxdat)
      CALL ef_get_coordinates(id, ARG1, Y_AXIS,
     .   arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), yaxdat)
      CALL ef_get_coordinates(id, ARG1, T_AXIS,
     .   arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), taxdat)

      i2 = arg_lo_ss(X_AXIS,ARG2)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      k2 = arg_lo_ss(Z_AXIS,ARG2)
      l2 = arg_lo_ss(T_AXIS,ARG2)

      i3 = arg_lo_ss(X_AXIS,ARG3)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)

      i4 = arg_lo_ss(X_AXIS,ARG4)
      j4 = arg_lo_ss(Y_AXIS,ARG4)
      k4 = arg_lo_ss(Z_AXIS,ARG4)
      l4 = arg_lo_ss(T_AXIS,ARG4)

*  Check to see if input x axis is modulo

      CALL ef_get_axis_info (id, ARG1, ax_name, ax_units, backward, 
     .                       modulo, regular)

      xlo = arg_lo_ss(X_AXIS,ARG1)
      xhi = arg_hi_ss(X_AXIS,ARG1)
      ylo = arg_lo_ss(Y_AXIS,ARG1)
      yhi = arg_hi_ss(Y_AXIS,ARG1)
      tlo = arg_lo_ss(T_AXIS,ARG1)
      thi = arg_hi_ss(T_AXIS,ARG1)

      IF ( modulo(1) ) THEN
         ic = STR_UPCASE (test_str, ax_units(1))
         ok = (test_str(1:3) .EQ. 'DEG' .OR. regular(1))
         IF (.NOT. ok) THEN
            err_msg = 'cannot handle MODULO axis that is not regular'
            go to 999
         ELSE
            dx = xaxdat(xlo+1,1,1,1) - xaxdat(xlo,1,1,1)
            delmodx = xaxdat(xhi,1,1,1) - xaxdat(xlo,1,1,1) + dx
            xaxdat(xhi+1,1,1,1) = xaxdat(xlo,1,1,1)
            IF (ok) delmodx = 360.
         ENDIF
      ENDIF
      IF ( modulo(4) ) THEN
         ic = STR_UPCASE (test_str, ax_units(4))
         ok = (regular(4))
         IF (.NOT. ok) THEN
            err_msg = 'cannot handle MODULO T axis that is not regular'
            go to 999
         ELSE
            dt = taxdat(tlo+1,1,1,1) - taxdat(tlo,1,1,1)
            delmodt = taxdat(thi,1,1,1) - taxdat(tlo,1,1,1) + dt
            taxdat(thi+1,1,1,1) = taxdat(tlo,1,1,1)
         ENDIF
      ENDIF

*  For each (xpt,ypt,tpt) triple, 
*  1) search the time coordinates of arg 1 for the nearest Time match
*  2) search the data array arg_1 for the nearest higher (x,y) grid 
*     coordinates.  Interpolate in 2 directions for the result.

      i2 = arg_lo_ss(X_AXIS,ARG2)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      k2 = arg_lo_ss(Z_AXIS,ARG2)
      l2 = arg_lo_ss(T_AXIS,ARG2)

      i3 = arg_lo_ss(X_AXIS,ARG3)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)

      j = res_lo_ss(Y_AXIS)
      l = res_lo_ss(T_AXIS)
      DO 500 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         ibot = ef_unspecified_int4	! Check if xpt points in xax range.
         imatch = 0

	 xpt = arg_2(i2,j2,k2,l2)
	 ypt = arg_3(i3,j3,k3,l3)
	 tpt = arg_4(i3,j3,k3,l3)

         IF (xpt .EQ. bad_flag(ARG2) .OR. ypt .EQ. bad_flag(ARG3) 
     .       .OR. tpt .EQ. bad_flag(ARG4) ) THEN
            DO 700 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               result(i,j,k,l) = bad_flag_result
  700       CONTINUE

         ELSE
* Find location of TPT in grid
            Lbot = ef_unspecified_int4	! Check if tpt point in tax range.
            Lmatch = 0
            DO 600 i4 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(t_AXIS,ARG1)
               IF (tpt .GE. taxdat(i4,1,1,1)) Lbot = i4
               IF (tpt .EQ. taxdat(i4,1,1,1)) Lmatch = i4

 	       IF (modulo(4)) THEN
	          DO WHILE (tpt .GE. taxdat(thi,1,1,1) ) 
                     tpt = tpt - delmodt
	          ENDDO
	          DO WHILE (tpt .LT. taxdat(tlo,1,1,1) )
		     tpt = tpt + delmodt
	          ENDDO
	       ENDIF

  600       CONTINUE
            IF (Lbot .EQ. ef_unspecified_int4) THEN
               DO 800 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
                  result(i,j,k,l) = bad_flag_result
  800          CONTINUE
               GOTO 400
            ELSE

               IF (lmatch .NE. 0) then
                  ltop = lbot
               ELSE
                  ltop = lbot + 1
               ENDIF
               IF (lbot .EQ. ef_unspecified_int4) ltop = lbot

               tbot = taxdat(lbot,1,1,1)
               ttop = taxdat(ltop,1,1,1)

               IF (ltop .GE. arg_hi_ss(T_AXIS,ARG1)) 
     .               ltop = arg_hi_ss(T_AXIS,ARG1)
               IF (ttop .LT. tbot) THEN
                  lbot = arg_hi_ss(T_AXIS,ARG1)
                  ltop = arg_lo_ss(T_AXIS,ARG1)
                  tbot = tbot - delmodt
                  IF (tpt .GT. ttop) tpt = tpt - delmodt
               ENDIF
               
               IF (ttop .EQ. tbot) THEN 
                  tfrac_lo = 1.
               ELSE
                  tfrac_lo = (ttop - tpt )/ (ttop - tbot)
               ENDIF
               tfrac_hi = 1. - tfrac_lo
 
            ENDIF
            
            DO 100 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
               IF (xpt .GE. xaxdat(i1,1,1,1)) ibot = i1
               IF (xpt .EQ. xaxdat(i1,1,1,1)) imatch = i1

cbf      may be some derivation from the exact value can be allowed:
cbf      if (xpt - xaxdat(i1,1,1,1)).le.eps)   imatch = i1

*  Locate the X point within the range of modulo X axis 

 	       IF (modulo(1)) THEN
	          DO WHILE (xpt .GE. xaxdat(xhi,1,1,1) ) 
                     xpt = xpt - delmodx
	          ENDDO
	          DO WHILE (xpt .LT. xaxdat(xlo,1,1,1) )
		     xpt = xpt + delmodx
	          ENDDO
	       ENDIF

  100       CONTINUE
            i1 = arg_hi_ss(X_AXIS,ARG1)
            IF (xpt .GT. xaxdat(i1,1,1,1)) THEN
               IF (.NOT. modulo(1)) 
     .             ibot = ef_unspecified_int4		! ARG_2 XPT outside of range 
		           				! (non modulo)
            ENDIF

cbf       for matching the next neighbour is not of interest
            IF (imatch .NE. 0) then
               itop = ibot
            ELSE
               itop = ibot + 1
            ENDIF

            IF (ibot .EQ. ef_unspecified_int4) THEN
               itop = ibot
            ELSE

               xbot = xaxdat(ibot,1,1,1)
               xtop = xaxdat(itop,1,1,1)
c ---
               IF (modulo(1) .AND. xtop .LT. xbot) THEN
                  ibot = arg_hi_ss(X_AXIS,ARG1)
                  itop = arg_lo_ss(X_AXIS,ARG1)
                  xbot = xbot - delmodx
                  IF (xpt .GT. xtop) xpt = xpt - delmodx
               ENDIF
c ---
            ENDIF


cbf        analogously in y direction
            jbot = ef_unspecified_int4	! Check if ypt points in yax range.
            jmatch = 0

   	    ypt = arg_3(i3,j3,k3,l3)
            DO 200 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
               IF (ypt .GE. yaxdat(j1,1,1,1) ) jbot = j1
            IF (ypt .EQ. yaxdat(j1,1,1,1) ) jmatch = j1
cbf         IF (ypt - yaxdat(j1,1,1,1) ) .LE. eps) jmatch = j1

  200       CONTINUE
            j1 = arg_hi_ss(Y_AXIS,ARG1)
            IF (arg_3(i3,j3,k3,l3) .GE. yaxdat(j1,1,1,1) ) THEN
                jbot = ef_unspecified_int4		! ARG_3 YPT outside of range 
						        ! (non modulo)

            ENDIF

            IF (jmatch .NE. 0) then
                jtop = jbot
            ELSE
               jtop = jbot + 1
            ENDIF
            IF (jbot .EQ. ef_unspecified_int4) jtop = jbot
            IF (jtop .GE. arg_hi_ss(Y_AXIS,ARG1)) 
     .                           jtop = arg_hi_ss(Y_AXIS,ARG1)        !10/2001 acm

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 333 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               l1 = arg_lo_ss(T_AXIS,ARG1)
               l2 = arg_lo_ss(T_AXIS,ARG2)
               l3 = arg_lo_ss(T_AXIS,ARG3)

*  First interpolate in x, getting values of the fcn at (x,jbot) and (x,jtop)

               IF (ibot .EQ. ef_unspecified_int4  .OR.  
     .		      jbot .EQ. ef_unspecified_int4) THEN
                  result(i,j,k,l) = bad_flag_result
               ELSE 

                  aa = arg_1(ibot,jbot,k1,lbot)
                  bb = arg_1(ibot,jbot,k1,ltop)
                  fbb = aa* tfrac_lo + bb* tfrac_hi

                  aa = arg_1(itop,jbot,k1,lbot)
                  bb = arg_1(itop,jbot,k1,ltop)
                  ftb = aa* tfrac_lo + bb* tfrac_hi

                  aa = arg_1(ibot,jtop,k1,lbot)
                  bb = arg_1(ibot,jtop,k1,ltop)
                  fbt = aa* tfrac_lo + bb* tfrac_hi

                  aa = arg_1(itop,jtop,k1,lbot)
                  bb = arg_1(itop,jtop,k1,ltop)
                  ftt = aa* tfrac_lo + bb* tfrac_hi

                  IF (fbb .NE. bad_flag(ARG1)  .AND.
     .                ftb .NE. bad_flag(ARG1)  .AND.
     .                fbt .NE. bad_flag(ARG1)  .AND.
     .                ftt .NE. bad_flag(ARG1) ) THEN

cbf    for matching x-axis no interpolation is need
                     IF (imatch .NE. 0) THEN
                        fxbot = fbb
                        fxtop = fbt
                     ELSE
                        frac = (xpt - xbot )/ (xtop - xbot)
                        fxbot = fbb + frac* (ftb - fbb)
                        fxtop = fbt + frac* (ftt - fbt)
                     ENDIF

*  Now interpolate in y, getting value at (x,y)

                     IF (jbot .GE. arg_lo_ss(Y_AXIS,ARG1)  .AND.
     .                   jtop .LE. arg_hi_ss(Y_AXIS,ARG1) ) THEN
                        ybot = yaxdat(jbot,1,1,1)
                        ytop = yaxdat(jtop,1,1,1)

                        IF (jmatch .NE. 0) THEN
                             result(i,j,k,l) = fxbot
                        ELSE
                            frac = (ypt - ybot)/ (ytop-ybot)
                            result(i,j,k,l) = fxbot + frac* 
     .                                         (fxtop - fxbot)
                        ENDIF
                     ELSE
                        result(i,j,k,l) = bad_flag_result
                     ENDIF

                  ELSE
                     result(i,j,k,l) = bad_flag_result
                  ENDIF	   ! fbb,ftp, etc not bad flags

	       ENDIF  ! itop, jtop not ef_unspecified_int4

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 333        CONTINUE

         ENDIF  ! xpt and ypt not bad values
 400     CONTINUE
         i2 = i2 + arg_incr(X_AXIS,ARG2)
         j2 = j2 + arg_incr(Y_AXIS,ARG2)
         k2 = k2 + arg_incr(Z_AXIS,ARG2)
         l2 = l2 + arg_incr(T_AXIS,ARG2)

         i3 = i3 + arg_incr(X_AXIS,ARG3)
         j3 = j3 + arg_incr(Y_AXIS,ARG3)
         k3 = k3 + arg_incr(Z_AXIS,ARG3)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
         j = j + res_incr(Y_AXIS)
500   CONTINUE


      RETURN
 999  CALL ef_bail_out (id, err_msg)

      END
