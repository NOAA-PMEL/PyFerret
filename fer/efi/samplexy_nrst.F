*
* samplexy_nrst.F
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
*
*   Version of samplexy to get data at the nearest grid point rather
*   than doing interpolation.
*
* This function samples 4-d data on the x and y axes indicated by args 2 and 3
* 9/10/99  Ansley Manke 
*         Result is abstract on the x axis, normal on the y axis,
*         and keeps the z and t axes of the input 4-d data.
*
* 1/13/2000 acm -- fix so that xpts and/or ypts can be on any axis.
* 3/ 1/2000 ACM -- fix bug in the definition of the work arrays
* 6/ 8/2000 ACM -- fix bug in incrementing k2, k3, l2, l3
*
* 7/2000  acm -- merge Bernadettes and increment fix into this version
* 5/2001  acm -- allow modulo axes
* 10/2001 acm -- check that jtop (box top) is within range of ARG1 y axis.
* 10/2001 acm -- check that jtop (box top) is within range of ARG1 y axis.
*  2/2001 acm -- check that modulo axes are also regular, and fix delmodx, delmody
*  2/2002 acm -- check xpt and ypt (x and y points to sample) that values arent missing.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
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


      SUBROUTINE samplexy_nrst_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Returns data sampled at a set of (X,Y) points, ',
     .        'using nearest grid intersection')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT, NORMAL,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 6)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'DAT_TO_SAMPLE')
      CALL ef_set_arg_desc(id, arg, 'variable (x,y,z,t,e,f) to sample')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, YES, YES, YES, YES)

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

      SUBROUTINE samplexy_nrst_result_limits(id)

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
*     2nd argument, to set the abstract axis lo and hi indices.

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
      SUBROUTINE samplexy_nrst_work_size(id)

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
      INTEGER mxh, myh, mxl, myl
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

*  Allocate the length of the axes for REAL*8 work arrays. Use the argument
*  subscript limits, so subsets are handled correctly.

*  xaxdat
      CALL ef_set_work_array_dims_6d(id, 1,
     .                              arg_lo_ss(X_AXIS,ARG1), 1, 1, 1, 1, 1,
     .                              arg_hi_ss(X_AXIS,ARG1), 1, 1, 1, 1, 1)

*  yaxdat
      CALL ef_set_work_array_dims_6d(id, 2,
     .                              arg_lo_ss(Y_AXIS,ARG1), 1, 1, 1, 1, 1,
     .                              arg_hi_ss(Y_AXIS,ARG1), 1, 1, 1, 1, 1)

*  xboxlo
      CALL ef_set_work_array_dims_6d(id, 3,
     .                              arg_lo_ss(X_AXIS,ARG1), 1, 1, 1, 1, 1,
     .                              arg_hi_ss(X_AXIS,ARG1), 1, 1, 1, 1, 1)

*  xboxhi
      CALL ef_set_work_array_dims_6d(id, 4,
     .                              arg_lo_ss(X_AXIS,ARG1), 1, 1, 1, 1, 1,
     .                              arg_hi_ss(X_AXIS,ARG1), 1, 1, 1, 1, 1)

*  yboxlo
      CALL ef_set_work_array_dims_6d(id, 5,
     .                              arg_lo_ss(Y_AXIS,ARG1), 1, 1, 1, 1, 1,
     .                              arg_hi_ss(Y_AXIS,ARG1), 1, 1, 1, 1, 1)

*  yboxhi
      CALL ef_set_work_array_dims_6d(id, 6,
     .                              arg_lo_ss(Y_AXIS,ARG1), 1, 1, 1, 1, 1,
     .                              arg_hi_ss(Y_AXIS,ARG1), 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
*
* In this subroutine we compute the result
*
      SUBROUTINE samplexy_nrst_compute(id, arg_1, arg_2, arg_3,
     .        result, xaxdat, yaxdat, xboxlo, xboxhi, yboxlo, yboxhi)

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

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

      REAL*8 xaxdat(wrk1lox:wrk1hix,wrk1loy:wrk1hiy,
     .              wrk1loz:wrk1hiz, wrk1lot:wrk1hit,
     .              wrk1loe:wrk1hie, wrk1lof:wrk1hif)
      REAL*8 yaxdat(wrk2lox:wrk2hix,wrk2loy:wrk2hiy,
     .              wrk2loz:wrk2hiz, wrk2lot:wrk2hit,
     .              wrk2loe:wrk2hie, wrk2lof:wrk2hif)
      REAL*8 xboxlo(wrk3lox:wrk3hix,wrk3loy:wrk3hiy,
     .              wrk3loz:wrk3hiz, wrk3lot:wrk3hit,
     .              wrk3loe:wrk3hie, wrk3lof:wrk3hif)
      REAL*8 xboxhi(wrk4lox:wrk4hix,wrk4loy:wrk4hiy,
     .              wrk4loz:wrk4hiz, wrk4lot:wrk4hit,
     .              wrk4loe:wrk4hie, wrk4lof:wrk4hif)
      REAL*8 yboxlo(wrk5lox:wrk5hix,wrk5loy:wrk5hiy,
     .              wrk5loz:wrk5hiz, wrk5lot:wrk5hit,
     .              wrk5loe:wrk5hie, wrk5lof:wrk5hif)
      REAL*8 yboxhi(wrk6lox:wrk6hix,wrk6loy:wrk6hiy,
     .              wrk6loz:wrk6hiz, wrk6lot:wrk6hit,
     .              wrk6loe:wrk6hie, wrk6lof:wrk6hif)

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

      LOGICAL TM_FPEQ, full_span, ok, its_dsg
      INTEGER STR_UPCASE

      INTEGER nx, nxx, nxy, nxz, nxt, nxe, nxf
      INTEGER ny, nyx, nyy, nyz, nyt, nye, nyf
      INTEGER ndimx, ndimy
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER ic, imatch, jmatch
      INTEGER ibot, jbot, itop, jtop, ihi, ilo, jhi, jlo, xflag
      REAL xbot, xtop, ybot, ytop
      REAL x_to_xbot, x_to_xtop, y_to_ybot, y_to_ytop
      REAL dx, dxlo, dxhi, dy, dylo, dyhi, delmodx, xpt, ypt
      CHARACTER err_msg*255, test_str*16

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), modulo(6), regular(6)
      INTEGER xlo, xhi, ylo, yhi
      

* Does the variable come from a Discrete Sampling Geometries dataset?  

      CALL EF_GET_ITS_DSG(id, ARG1, its_dsg )
      IF (its_dsg) THEN
         err_msg = 
     . 'Data is from a Discrete Sampling Geometries (DSG) dataset. Function does not apply.'
       GOTO 999
      ENDIF


      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      nxx = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1
      nxy = arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1
      nxz = arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2) + 1
      nxt = arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1
      nxe = arg_hi_ss(E_AXIS,ARG2) - arg_lo_ss(E_AXIS,ARG2) + 1
      nxf = arg_hi_ss(F_AXIS,ARG2) - arg_lo_ss(F_AXIS,ARG2) + 1
      nx = max(nxx, nxy, nxz, nxt, nxe, nxf)

      nyx = arg_hi_ss(X_AXIS,ARG3) - arg_lo_ss(X_AXIS,ARG3) + 1
      nyy = arg_hi_ss(Y_AXIS,ARG3) - arg_lo_ss(Y_AXIS,ARG3) + 1
      nyz = arg_hi_ss(Z_AXIS,ARG3) - arg_lo_ss(Z_AXIS,ARG3) + 1
      nyt = arg_hi_ss(T_AXIS,ARG3) - arg_lo_ss(T_AXIS,ARG3) + 1
      nye = arg_hi_ss(E_AXIS,ARG3) - arg_lo_ss(E_AXIS,ARG3) + 1
      nyf = arg_hi_ss(F_AXIS,ARG3) - arg_lo_ss(F_AXIS,ARG3) + 1
      ny = max(nyx, nyy, nyz, nyt, nye, nyf)

      ndimx = 0
      ndimy = 0
      DO i = X_AXIS,F_AXIS
         IF ( arg_hi_ss(i,ARG2) .GT. arg_lo_ss(i,ARG2) )
     .      ndimx = ndimx + 1
         IF ( arg_hi_ss(i,ARG3) .GT. arg_lo_ss(i,ARG3) ) 
     .      ndimy = ndimy + 1
      ENDDO

      IF ( (nx .NE. ny) .OR. (ndimx .GT. 1) .OR. (ndimy .GT. 1) ) THEN
         err_msg =
     .  'Arguments 2 and 3 must be 1-dimensional lists of equal length'
         GO TO 999
      ENDIF

*  Get x and y coordinates of the data to be sampled.

      ilo = arg_lo_ss(X_AXIS,ARG1)
      ihi = arg_hi_ss(X_AXIS,ARG1)
      jlo = arg_lo_ss(Y_AXIS,ARG1)
      jhi = arg_hi_ss(Y_AXIS,ARG1)

      CALL ef_get_coordinates(id, ARG1, X_AXIS, ilo, ihi, xaxdat)
      CALL ef_get_coordinates(id, ARG1, Y_AXIS, jlo, jhi, yaxdat)

*  Check to see if input x axis is modulo, and if axes are regular.

      CALL ef_get_axis_info_6d(id, ARG1, ax_name, ax_units,
     .                         backward, modulo, regular)
      IF ( regular(1) ) THEN
         dx = xaxdat(ilo+1,1,1,1,1,1) - xaxdat(ilo,1,1,1,1,1)
         dxlo = dx/2.
         dxhi = dx/2.
      ELSE
         CALL ef_get_box_limits(id, ARG1, X_AXIS, ilo, ihi,
     .                          xboxlo, xboxhi)
         dxlo = (xboxlo(ilo+1,1,1,1,1,1) - xboxlo(ilo,1,1,1,1,1) ) / 2.
         dxhi = (xboxhi(ihi,1,1,1,1,1) - xboxhi(ihi-1,1,1,1,1,1) ) / 2.
      ENDIF

      IF ( regular(2) ) THEN
         dy = yaxdat(jlo+1,1,1,1,1,1) - yaxdat(jlo,1,1,1,1,1)
         dylo = dy/2.
         dyhi = dy/2.
      ELSE
         CALL ef_get_box_limits(id, ARG1, Y_AXIS, jlo, jhi,
     .                          yboxlo, yboxhi)
         dylo = (yboxlo(jlo+1,1,1,1,1,1) - yboxlo(jlo,1,1,1,1,1) )/ 2.
         dyhi = (yboxhi(jhi,1,1,1,1,1) - yboxlo(jhi-1,1,1,1,1,1) )/ 2.
      ENDIF

* Check to see if input x,y axis is modulo. If its a longitude 
* axis, check for full-span

      IF ( modulo(1) ) THEN
         ic = STR_UPCASE (test_str, ax_units(1))
         ok = (test_str(1:3) .EQ. 'DEG') 
         dx = xaxdat(ilo+1,1,1,1,1,1) - xaxdat(ilo,1,1,1,1,1)
         delmodx = (xaxdat(ihi,1,1,1,1,1) + dxhi) -
     .             (xaxdat(ilo,1,1,1,1,1) - dxlo)
         IF ( .NOT. regular(1) )
     .      delmodx = xboxhi(ihi,1,1,1,1,1) - xboxlo(ilo,1,1,1,1,1)
         IF ( ok ) THEN 
            full_span = (delmodx .GT. 360.-dx)
            delmodx = 360.
         ELSE
            full_span = .TRUE.
         ENDIF
      ENDIF

*  For each (xpt,ypt) pair, search the data array 
*   arg_1 for the nearest higher (x,y) grid coordinates.  Find the 
*   nearest in 2 directions for the result.


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

* Y-axis is normal to the result
      j = res_lo_ss(Y_AXIS)

      DO 700 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         xpt = arg_2(i2,j2,k2,l2,m2,n2)
         ypt = arg_3(i3,j3,k3,l3,m3,n3)
         IF (xpt .EQ. bad_flag(ARG2) .OR. ypt .EQ. bad_flag(ARG3)) THEN
            DO n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
            ENDDO
            ENDDO
            ENDDO
            ENDDO
*           Go to the next pair of coordinates
            GOTO 690
         ENDIF

* Flag for outside of a modulo axis
         xflag = 0

* Check if xpt is in the x-axis range.
         ibot = ef_unspecified_int4
         itop = ef_unspecified_int4
         imatch = ef_unspecified_int4
*  Locate the X point within the range of modulo X axis 
*  If we have the whole span, then use the point at the other
*  end to handle points within a half-grid cell of the edge.

         IF ( modulo(1) .AND. full_span ) THEN
            IF ( xpt .GE. xaxdat(ihi,1,1,1,1,1) .AND.
     .           xpt .LE. xaxdat(ihi,1,1,1,1,1) + dxhi ) THEN
               ibot = ihi
               xflag = 1
            ELSE IF ( xpt .GE. xaxdat(ihi,1,1,1,1,1) ) THEN
               DO WHILE ( xpt .GE. xaxdat(ihi,1,1,1,1,1) ) 
                  xpt = xpt - delmodx
               ENDDO
            ENDIF

            IF ( xpt .LT. xaxdat(ilo,1,1,1,1,1) .AND.
     .           xpt .GE. xaxdat(ilo,1,1,1,1,1) - dxlo ) THEN
               itop = ilo
               xflag = -1
            ELSE IF ( xpt .LT. xaxdat(ilo,1,1,1,1,1) ) THEN
               DO WHILE (xpt .LT. xaxdat(ilo,1,1,1,1,1) )
                  xpt = xpt + delmodx
               ENDDO
            ENDIF
	 ENDIF

* Source variable has subspan-modulo axis 

         IF ( modulo(1) .AND. .NOT.full_span ) THEN
            IF (modulo(1) .AND. xpt .GE. xaxdat(ihi,1,1,1,1,1) ) THEN
               DO WHILE ( xpt .GE. xaxdat(ihi,1,1,1,1,1) ) 
               xpt = xpt - delmodx
               ENDDO
            ENDIF

	    IF (modulo(1) .AND. xpt .LT. xaxdat(ilo,1,1,1,1,1) ) THEN
               DO WHILE (xpt .LT. xaxdat(ilo,1,1,1,1,1) )
                  xpt = xpt + delmodx
               ENDDO
            ENDIF
         ENDIF

         DO i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
            IF ( TM_FPEQ(xpt, xaxdat(i1,1,1,1,1,1)) ) THEN
               imatch = i1
               GOTO 130
            ELSE IF ( i1  .EQ. arg_lo_ss(X_AXIS,ARG1)     .AND.
     .                xpt .LT. xaxdat(i1,1,1,1,1,1)       .AND.
     .                xpt .GT. xaxdat(i1,1,1,1,1,1)-dxhi  .AND.
     .               .NOT. modulo(1) ) THEN
               ibot = i1
            ELSE IF ( xpt .GE. xaxdat(i1,1,1,1,1,1) ) THEN
               ibot = i1
            ENDIF
         ENDDO

         IF ( .NOT. modulo(1) .OR. (modulo(1) .AND..NOT.full_span )) THEN
            IF ( xpt .GT. xaxdat(ihi,1,1,1,1,1) .AND.
     .           xpt .LT. xaxdat(ihi,1,1,1,1,1) + dxhi ) THEN
               ibot = ihi
            ELSE IF ( xpt .GT. xaxdat(ihi,1,1,1,1,1) ) THEN
               ibot = ef_unspecified_int4   ! ARG_2 XPT outside of range 
            ENDIF
         ENDIF

  130    IF ( imatch .NE. ef_unspecified_int4 ) THEN
            ibot = imatch
            itop = imatch
         ELSE IF ( ibot .NE. ef_unspecified_int4 ) THEN
            itop = ibot + 1
            IF ( itop .GT. ihi ) itop = ihi
         ENDIF

* If the data is at the very top or bottom of a full-span modulo x axis, get the
* point from the other end.

         IF ( xflag .EQ. 1 ) THEN
            ibot = ihi
            itop = ilo
         ENDIF
         IF ( xflag .EQ. -1 ) THEN
            itop = ilo
            ibot = ihi
         ENDIF

         IF ( ibot .EQ. ef_unspecified_int4 ) THEN
*           Coordinate outside the grid
            DO n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            DO k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
            ENDDO
            ENDDO
            ENDDO
            ENDDO
*           Go to the next pair of coordinates
            GOTO 690
         ENDIF

	 IF ( imatch .NE. ef_unspecified_int4 ) THEN
cbf         matching the next neighbour is not of interest
            i1 = imatch
         ELSE IF ( ibot .EQ. ef_unspecified_int4 ) THEN
* xpt not in the the X-axis range
            i1 = ef_unspecified_int4
         ELSE
* ibot will be less than arg_hi_ss(X_AXIS,ARG1) because of earlier check
            xbot = xaxdat(ibot,  1,1,1,1,1)
            xtop = xaxdat(ibot+1,1,1,1,1,1)
            x_to_xbot = ABS(xpt - xbot)
            x_to_xtop = ABS(xpt - xtop)
            IF (x_to_xbot .LE. x_to_xtop) THEN
               i1 = ibot
            ELSE
               i1 = ibot+1
            ENDIF
         ENDIF

cbf      analogously in y direction

         jbot = ef_unspecified_int4
         jmatch = ef_unspecified_int4

* Check if ypt exceeds the Y-axis range.
         j1 = arg_hi_ss(Y_AXIS,ARG1)
         IF ( TM_FPEQ(yaxdat(j1,1,1,1,1,1), ypt) ) THEN
            jmatch = j1
            GOTO 220
         ELSE IF ( ypt .GT. yaxdat(j1,1,1,1,1,1) ) THEN
            GOTO 220
         ENDIF

* Find ypt in the Y-axis range.
         DO j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
            IF ( TM_FPEQ(yaxdat(j1,1,1,1,1,1), ypt) ) THEN
               jmatch = j1
               GOTO 220
            ELSE IF ( ypt .GT. yaxdat(j1,1,1,1,1,1) ) THEN
               jbot = j1
            ENDIF
         ENDDO

  220    IF ( jmatch .NE. ef_unspecified_int4 ) THEN
cbf         matching the next neighbour is not of interest
            j1 = jmatch
         ELSE IF ( jbot .EQ. ef_unspecified_int4 ) THEN
* ypt not in the the Y-axis range
            j1 = ef_unspecified_int4
         ELSE
* jbot will be less than arg_hi_ss(Y_AXIS,ARG1) because of earlier check
            ybot = yaxdat(jbot,  1,1,1,1,1)
            ytop = yaxdat(jbot+1,1,1,1,1,1)
            y_to_ybot = abs(ypt - ybot)
            y_to_ytop = abs(ypt - ytop)
            IF (y_to_ybot .LE. y_to_ytop) THEN
               j1 = jbot
            ELSE
               j1 = jbot+1
            ENDIF
         ENDIF

* Loop through the Z/T/E/F axes, assigning results for this coordinate pair
         n1 = arg_lo_ss(F_AXIS,ARG1)
         DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

         m1 = arg_lo_ss(E_AXIS,ARG1)
         DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

         l1 = arg_lo_ss(T_AXIS,ARG1)
         DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               IF ( i1 .EQ. ef_unspecified_int4  .OR.  
     .              j1 .EQ. ef_unspecified_int4 ) THEN
*                 coordinates specified are outside the grid
                  result(i,j,k,l,m,n) = bad_flag_result
               ELSE IF ( arg_1(i1,j1,k1,l1,m1,n1) 
     .                         .EQ. bad_flag(ARG1) ) THEN
*                 closest grid point has an undefined value
                  result(i,j,k,l,m,n) = bad_flag_result
               ELSE
                  result(i,j,k,l,m,n) = arg_1(i1,j1,k1,l1,m1,n1)
               ENDIF

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300        CONTINUE

            l1 = l1 + arg_incr(T_AXIS,ARG1)
 400     CONTINUE

            m1 = m1 + arg_incr(E_AXIS,ARG1)
 500     CONTINUE

            n1 = n1 + arg_incr(F_AXIS,ARG1)
 600     CONTINUE

* Only one of the ARG2 and one of the ARG3 actually increments
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
 700  CONTINUE

      RETURN

 999  CALL ef_bail_out (id, err_msg)

      END
