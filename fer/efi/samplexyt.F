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

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled at a set of (X,Y,T) points, ',
     .        'using linear interpolation')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 4)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT,        NORMAL,
     .                                IMPLIED_BY_ARGS, NORMAL,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 5)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'variable (x,y,z,t,e,f) to sample')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, YES, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'XPTS')
      CALL ef_set_arg_desc(id, arg, 'X values of sample points')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'YPTS')
      CALL ef_set_arg_desc(id, arg, 'Y values of sample points')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg, 'T values of sample points')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

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

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER my_lo, my_hi
      INTEGER nx, ny, nz, nt, ne, nf

*     Use utility functions to get context information about the 
*     second argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG2) - arg_lo_ss(X_AXIS, ARG2) + 1
      ny = arg_hi_ss(Y_AXIS, ARG2) - arg_lo_ss(Y_AXIS, ARG2) + 1
      nz = arg_hi_ss(Z_AXIS, ARG2) - arg_lo_ss(Z_AXIS, ARG2) + 1
      nt = arg_hi_ss(T_AXIS, ARG2) - arg_lo_ss(T_AXIS, ARG2) + 1
      ne = arg_hi_ss(E_AXIS, ARG2) - arg_lo_ss(E_AXIS, ARG2) + 1
      nf = arg_hi_ss(F_AXIS, ARG2) - arg_lo_ss(F_AXIS, ARG2) + 1

      my_lo = 1
      my_hi = max(nx,ny,nz,nt,ne,nf)

      CALL ef_set_axis_limits(id, X_AXIS, my_lo, my_hi)

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

      INTEGER mxh, myh, mth, mxl, myl, mtl
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

* Allocate double the length of the axes for REAL*8 work arrays.
* For x and t axes allow yet one more point for handling modulo axes

      mxl = arg_lo_ss(X_AXIS,ARG1)
      myl = arg_lo_ss(Y_AXIS,ARG1)
      mtl = arg_lo_ss(T_AXIS,ARG1)

      mxh = mxl + 
     .     2* (arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 2)
      myh = myl + 
     .     2* (arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1) + 1)
      mth = mtl + 
     .     2* (arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1) + 2)

*  xaxdat
      CALL ef_set_work_array_dims_6d(id, 1,
     .                              mxl, 1, 1, 1, 1, 1,
     .                              mxh, 1, 1, 1, 1, 1)

*  yaxdat
      CALL ef_set_work_array_dims_6d(id, 2,
     .                              myl, 1, 1, 1, 1, 1,
     .                              myh, 1, 1, 1, 1, 1)

*  taxdat
      CALL ef_set_work_array_dims_6d(id, 3,
     .                              mtl, 1, 1, 1, 1, 1,
     .                              mth, 1, 1, 1, 1, 1)

*  taxdatlo
      CALL ef_set_work_array_dims_6d(id, 4,
     .                              mtl, 1, 1, 1, 1, 1,
     .                              mth, 1, 1, 1, 1, 1)

*  taxdathi
      CALL ef_set_work_array_dims_6d(id, 5,
     .                              mtl, 1, 1, 1, 1, 1,
     .                              mth, 1, 1, 1, 1, 1)

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
     .         result, xaxdat, yaxdat, taxdat, taxdatlo, taxdathi)

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

* Ignore the Y/Z/T/E/F dimensions in the work arrays since
* they are not used and Fortran is column major
      REAL*8 xaxdat(wrk1lox:wrk1lox+(wrk1hix-wrk1lox)/2)
      REAL*8 yaxdat(wrk2lox:wrk2lox+(wrk2hix-wrk2lox)/2)
      REAL*8 taxdat(wrk3lox:wrk3lox+(wrk3hix-wrk3lox)/2)
      REAL*8 taxdatlo(wrk4lox:wrk4lox+(wrk4hix-wrk4lox)/2)
      REAL*8 taxdathi(wrk5lox:wrk5lox+(wrk5hix-wrk5lox)/2)

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

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER STR_UPCASE
      LOGICAL TM_FPEQ

      INTEGER nx, nxx, nxy, nxz, nxt, nxe, nxf
      INTEGER ny, nyx, nyy, nyz, nyt, nye, nyf
      INTEGER nt, ntx, nty, ntz, ntt, nte, ntf
      INTEGER ndimx, ndimy, ndimt
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER i4, j4, k4, l4, m4, n4
      INTEGER ic, imatch, jmatch, lmatch
      INTEGER ibot, itop, jbot, jtop, lbot, ltop
      INTEGER xlo, xhi, ylo, yhi, tlo, thi
      LOGICAL ok
      REAL fxbot, fxtop, fbb, ftb, fbt, ftt, aa, bb
      REAL xbot, xtop, ybot, ytop, tbot, ttop
      REAL xfrac, yfrac, tfrac_lo, tfrac_hi
      REAL dx, dt, delmodx, delmodt, xpt, ypt, tpt
      CHARACTER err_msg*255, test_str*16

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), modulo(6), regular(6)

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      nxx = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      nxy = arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1
      nxz = arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2) + 1
      nxt = arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1
      nxe = arg_hi_ss(E_AXIS,ARG2) - arg_lo_ss(E_AXIS,ARG2) + 1
      nxf = arg_hi_ss(F_AXIS,ARG2) - arg_lo_ss(F_AXIS,ARG2) + 1
      nx = MAX(nxx, nxy, nxz, nxt, nxe, nxf)

      nyx = arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3) + 1
      nyy = arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3) + 1
      nyz = arg_hi_ss(Z_AXIS,ARG3) - arg_lo_ss(Z_AXIS,ARG3) + 1
      nyt = arg_hi_ss(T_AXIS,ARG3) - arg_lo_ss(T_AXIS,ARG3) + 1
      nye = arg_hi_ss(E_AXIS,ARG3) - arg_lo_ss(E_AXIS,ARG3) + 1
      nyf = arg_hi_ss(F_AXIS,ARG3) - arg_lo_ss(F_AXIS,ARG3) + 1
      ny = MAX(nyx, nyy, nyz, nyt, nye, nyf)

      ntx = arg_hi_ss(X_AXIS,ARG4) - arg_lo_ss(X_AXIS,ARG4) + 1
      nty = arg_hi_ss(Y_AXIS,ARG4) - arg_lo_ss(Y_AXIS,ARG4) + 1
      ntz = arg_hi_ss(Z_AXIS,ARG4) - arg_lo_ss(Z_AXIS,ARG4) + 1
      ntt = arg_hi_ss(T_AXIS,ARG4) - arg_lo_ss(T_AXIS,ARG4) + 1
      nte = arg_hi_ss(E_AXIS,ARG4) - arg_lo_ss(E_AXIS,ARG4) + 1
      ntf = arg_hi_ss(F_AXIS,ARG4) - arg_lo_ss(F_AXIS,ARG4) + 1
      nt = MAX(ntx, nty, ntz, ntt, nte, ntf)

      ndimx = 0
      ndimy = 0
      ndimt = 0
      DO 10 i = X_AXIS,F_AXIS
         IF ( arg_hi_ss(i,ARG2) .GT. arg_lo_ss(i,ARG2) )
     .       ndimx = ndimx + 1
         IF ( arg_hi_ss(i,ARG3) .GT. arg_lo_ss(i,ARG3) )
     .       ndimy = ndimy + 1
         IF ( arg_hi_ss(i,ARG4) .GT. arg_lo_ss(i,ARG4) )
     .       ndimt = ndimt + 1
   10 CONTINUE

      IF ((nx .NE. ny) .OR. (nx .NE. nt) .OR. 
     .    (ndimx .GT. 1) .OR. (ndimy .GT. 1) .OR. (ndimt .GT. 1)) THEN
         WRITE (err_msg, 20)
         GO TO 999
      ENDIF

   20 FORMAT('Arguments 2, 3, and 4 must be 1-dimensional ',
     .       'lists of equal length')

*  Get x, y, t coordinates of the data to be sampled.

      CALL ef_get_coordinates(id, ARG1, X_AXIS,
     .   arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), xaxdat)
      CALL ef_get_coordinates(id, ARG1, Y_AXIS,
     .   arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), yaxdat)
      CALL ef_get_coordinates(id, ARG1, T_AXIS,
     .   arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), taxdat)
      CALL ef_get_box_lo_lim(id, ARG1, T_AXIS,
     .   arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), taxdatlo)
      CALL ef_get_box_hi_lim(id, ARG1, T_AXIS,
     .   arg_lo_ss(T_AXIS, ARG1), arg_hi_ss(T_AXIS, ARG1), taxdathi)

*  Check to see if input x axis is modulo

      CALL ef_get_axis_info_6d(id, ARG1, ax_name, ax_units,
     .                         backward, modulo, regular)

      xlo = arg_lo_ss(X_AXIS,ARG1)
      xhi = arg_hi_ss(X_AXIS,ARG1)
      ylo = arg_lo_ss(Y_AXIS,ARG1)
      yhi = arg_hi_ss(Y_AXIS,ARG1)
      tlo = arg_lo_ss(T_AXIS,ARG1)
      thi = arg_hi_ss(T_AXIS,ARG1)

      IF ( modulo(1) ) THEN
         ic = STR_UPCASE (test_str, ax_units(1))
         ok = (test_str(1:3) .EQ. 'DEG' .OR. regular(1))
         IF ( .NOT. ok ) THEN
            err_msg = 'cannot handle MODULO X axis that is not regular'
            GO TO 999
         ENDIF
         dx = xaxdat(xlo+1) - xaxdat(xlo)
         delmodx = 360.
         xaxdat(xhi+1) = xaxdat(xlo)
      ENDIF
      IF ( modulo(4) ) THEN
         IF ( .NOT. regular(4) ) THEN
            err_msg = 'cannot handle MODULO T axis that is not regular'
            GO TO 999
         ENDIF
         dt = taxdathi(tlo) - taxdatlo(tlo)
         delmodt = taxdat(thi) - taxdat(tlo) + dt
         taxdat(thi+1) = taxdat(tlo)
      ENDIF

*  For each (xpt,ypt,tpt) triple, 
*  1) search the time coordinates of arg 1 for the nearest Time match
*  2) search the data array arg_1 for the nearest higher (x,y) grid 
*     coordinates.  Interpolate in 2 directions for the result.

      i2 = arg_lo_ss(X_AXIS,ARG2)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      k2 = arg_lo_ss(Z_AXIS,ARG2)
      l2 = arg_lo_ss(T_AXIS,ARG2)
      m2 = arg_lo_ss(E_AXIS,ARG2)
      n2 = arg_lo_ss(F_AXIS,ARG2)

      i3 = arg_lo_ss(X_AXIS,ARG3)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      k3 = arg_lo_ss(Z_AXIS,ARG3)
      l3 = arg_lo_ss(T_AXIS,ARG3)
      m3 = arg_lo_ss(E_AXIS,ARG3)
      n3 = arg_lo_ss(F_AXIS,ARG3)

      i4 = arg_lo_ss(X_AXIS,ARG4)
      j4 = arg_lo_ss(Y_AXIS,ARG4)
      k4 = arg_lo_ss(Z_AXIS,ARG4)
      l4 = arg_lo_ss(T_AXIS,ARG4)
      m4 = arg_lo_ss(E_AXIS,ARG4)
      n4 = arg_lo_ss(F_AXIS,ARG4)

* Y-axis and T-axis is normal to the result
      j = res_lo_ss(Y_AXIS)
      l = res_lo_ss(T_AXIS)

      DO 700 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         xpt = arg_2(i2,j2,k2,l2,m2,n2)
         ypt = arg_3(i3,j3,k3,l3,m3,n3)
         tpt = arg_4(i4,j4,k4,l4,m4,n4)

         IF ( xpt .EQ. bad_flag(ARG2) .OR.
     .        ypt .EQ. bad_flag(ARG3) .OR.
     .        tpt .EQ. bad_flag(ARG4) ) THEN
            DO 30 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO 30 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO 30 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
   30       CONTINUE
*           Go to the next triplet of coordinates
            GOTO 690
         ENDIF

* Try to get tpt in range if the T-axis is modulo
         IF ( modulo(4) ) THEN
            DO WHILE ( tpt .GE. taxdathi(thi) ) 
               tpt = tpt - delmodt
            ENDDO
            DO WHILE ( tpt .LT. taxdatlo(tlo) )
               tpt = tpt + delmodt
            ENDDO
         ENDIF

* Check if tpt is in the T-axis range.
         lbot = ef_unspecified_int4
         lmatch = ef_unspecified_int4
         DO 410 l1 = arg_lo_ss(T_AXIS,ARG1), arg_hi_ss(T_AXIS,ARG1)
            IF ( TM_FPEQ(tpt, taxdat(l1)) ) THEN
               lmatch = l1
               GOTO 420
            ELSE IF ( tpt .GE. taxdatlo(l1) ) THEN
               lbot = l1
            ENDIF
  410    CONTINUE

* Check that tpt is not beyond the upper end of the T-axis range.
         IF ( tpt .GT. taxdathi(thi) ) THEN
            IF ( .NOT. modulo(4) )
     .         lbot = ef_unspecified_int4
         ENDIF

  420    IF ( lmatch .NE. ef_unspecified_int4 ) THEN
            lbot = lmatch
            ltop = lmatch
         ELSE IF ( lbot .NE. ef_unspecified_int4 ) THEN
*           using cell bounds taxdatlo, taxdathi
            ltop = lbot
         ELSE
*           tpt outside the grid
            DO 430 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO 430 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO 430 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
  430       CONTINUE
*           Go to the next triplet of coordinates
            GOTO 690
         ENDIF

         tbot = taxdatlo(lbot)
         ttop = taxdathi(ltop)

         IF ( modulo(4) .AND. (ttop .LT. tbot) ) THEN
            lbot = thi
            ltop = tlo
            tbot = tbot - delmodt
            IF ( tpt .GT. ttop )
     .         tpt = tpt - delmodt
         ENDIF
         IF ( ttop .EQ. tbot ) THEN 
            tfrac_lo = 1.
         ELSE
            tfrac_lo = (ttop - tpt) / (ttop - tbot)
         ENDIF
         tfrac_hi = 1. - tfrac_lo
 
* Try to get xpt in range if the X-axis is modulo
         IF ( modulo(1) ) THEN
            DO WHILE ( xpt .GE. xaxdat(xhi) ) 
               xpt = xpt - delmodx
            ENDDO
            DO WHILE ( xpt .LT. xaxdat(xlo) )
               xpt = xpt + delmodx
            ENDDO
         ENDIF

* Check if xpt is in the X-axis range.
         ibot = ef_unspecified_int4
         imatch = ef_unspecified_int4

         DO 110 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
            IF ( TM_FPEQ(xpt, xaxdat(i1)) ) THEN
               imatch = i1
               GOTO 120
            ELSE IF ( xpt .GT. xaxdat(i1) ) THEN
               ibot = i1
            ENDIF
  110    CONTINUE

* Check that xpt is not beyond the upper end of the X-axis range.
         IF ( xpt .GT. xaxdat(xhi) ) THEN
            IF ( .NOT. modulo(1) ) 
     .         ibot = ef_unspecified_int4
         ENDIF

  120    IF ( imatch .NE. ef_unspecified_int4 ) THEN
            ibot = imatch
            itop = imatch
         ELSE IF ( ibot .NE. ef_unspecified_int4 ) THEN
* If modulo(1) then there is an extra xaxdat so itop is in range
            itop = ibot + 1
         ELSE
*           xpt outside the grid
            DO 130 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO 130 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO 130 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
  130       CONTINUE
*           Go to the next triplet of coordinates
            GOTO 690
         ENDIF

         xbot = xaxdat(ibot)
         xtop = xaxdat(itop)
         IF ( modulo(1) .AND. (xtop .LT. xbot) ) THEN
            ibot = xhi
            itop = xlo
            xbot = xbot - delmodx
            IF (xpt .GT. xtop)
     .         xpt = xpt - delmodx
         ENDIF
         IF ( xtop .EQ. xbot ) THEN 
            xfrac = 0.
         ELSE
            xfrac = (xpt - xbot) / (xtop - xbot)
         ENDIF

* Check if ypt is in the Y-axis range.
         jbot = ef_unspecified_int4
         jmatch = ef_unspecified_int4

         DO 210 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
            IF ( TM_FPEQ(ypt, yaxdat(j1)) ) THEN
               jmatch = j1
               GOTO 220
            ELSE IF ( ypt .GT. yaxdat(j1) ) THEN
               jbot = j1
            ENDIF
  210    CONTINUE

* Check that ypt is not beyond the upper end of the Y-axis range.
         IF ( ypt .GT. yaxdat(yhi) ) THEN
             jbot = ef_unspecified_int4
         ENDIF

  220    IF ( jmatch .NE. ef_unspecified_int4 ) THEN
            jbot = jmatch
            jtop = jmatch
         ELSE IF ( jbot .NE. ef_unspecified_int4) THEN
            jtop = jbot + 1
         ELSE
*           ypt outside the grid
            DO 230 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO 230 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO 230 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
  230       CONTINUE
*           Go to the next triplet of coordinates
            GOTO 690
         ENDIF

         ybot = yaxdat(jbot)
         ytop = yaxdat(jtop)
         IF ( ytop .EQ. ybot ) THEN 
            yfrac = 0.
         ELSE
            yfrac = (ypt - ybot) / (ytop - ybot)
         ENDIF

* Loop through the Z/E/F axes, assigning results for this coordinate pair
         n1 = arg_lo_ss(F_AXIS,ARG1)
         DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         m1 = arg_lo_ss(E_AXIS,ARG1)
         DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               IF ( arg_1(ibot,jbot,k1,lbot,m1,n1)
     .                         .NE. bad_flag(ARG1) .AND.
     .              arg_1(ibot,jbot,k1,ltop,m1,n1)
     .                         .NE. bad_flag(ARG1) .AND.
     .              arg_1(ibot,jtop,k1,lbot,m1,n1)
     .                         .NE. bad_flag(ARG1) .AND.
     .              arg_1(ibot,jtop,k1,ltop,m1,n1)
     .                         .NE. bad_flag(ARG1) .AND.
     .              arg_1(itop,jbot,k1,lbot,m1,n1)
     .                         .NE. bad_flag(ARG1) .AND.
     .              arg_1(itop,jbot,k1,ltop,m1,n1)
     .                         .NE. bad_flag(ARG1) .AND.
     .              arg_1(itop,jtop,k1,lbot,m1,n1)
     .                         .NE. bad_flag(ARG1) .AND.
     .              arg_1(itop,jtop,k1,ltop,m1,n1)
     .                         .NE. bad_flag(ARG1) ) THEN

* First T-axis interpolation
                  aa = arg_1(ibot,jbot,k1,lbot,m1,n1)
                  bb = arg_1(ibot,jbot,k1,ltop,m1,n1)
                  fbb = aa * tfrac_lo + bb * tfrac_hi

                  aa = arg_1(itop,jbot,k1,lbot,m1,n1)
                  bb = arg_1(itop,jbot,k1,ltop,m1,n1)
                  ftb = aa * tfrac_lo + bb * tfrac_hi

                  aa = arg_1(ibot,jtop,k1,lbot,m1,n1)
                  bb = arg_1(ibot,jtop,k1,ltop,m1,n1)
                  fbt = aa * tfrac_lo + bb * tfrac_hi

                  aa = arg_1(itop,jtop,k1,lbot,m1,n1)
                  bb = arg_1(itop,jtop,k1,ltop,m1,n1)
                  ftt = aa * tfrac_lo + bb * tfrac_hi

* Next X axis interpolation
                  fxbot = fbb + xfrac * (ftb - fbb)
                  fxtop = fbt + xfrac * (ftt - fbt)

* Now Y-axis interpolation
                  result(i,j,k,l,m,n) = fxbot + yfrac *
     .                                  (fxtop - fxbot)

               ELSE

*                  some component is undefined
                  result(i,j,k,l,m,n) = bad_flag_result

               ENDIF

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300        CONTINUE

            m1 = m1 + arg_incr(E_AXIS,ARG1)
 500     CONTINUE

            n1 = n1 + arg_incr(F_AXIS,ARG1)
 600     CONTINUE

* Only one of the ARG2, one of the ARG3, and one of ARG4 actually increments
 690     i2 = i2 + arg_incr(X_AXIS,ARG2)
         j2 = j2 + arg_incr(Y_AXIS,ARG2)
         k2 = k2 + arg_incr(Z_AXIS,ARG2)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         m2 = m2 + arg_incr(E_AXIS,ARG2)
         n2 = n2 + arg_incr(F_AXIS,ARG2)

         i3 = i3 + arg_incr(X_AXIS,ARG3)
         j3 = j3 + arg_incr(Y_AXIS,ARG3)
         k3 = k3 + arg_incr(Z_AXIS,ARG3)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
         m3 = m3 + arg_incr(E_AXIS,ARG3)
         n3 = n3 + arg_incr(F_AXIS,ARG3)

         i4 = i4 + arg_incr(X_AXIS,ARG4)
         j4 = j4 + arg_incr(Y_AXIS,ARG4)
         k4 = k4 + arg_incr(Z_AXIS,ARG4)
         l4 = l4 + arg_incr(T_AXIS,ARG4)
         m4 = m4 + arg_incr(E_AXIS,ARG4)
         n4 = n4 + arg_incr(F_AXIS,ARG4)
  700 CONTINUE

      RETURN

  999 CALL ef_bail_out (id, err_msg)

      END
