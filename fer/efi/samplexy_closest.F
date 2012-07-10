*
* samplexy_closest.F
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


      SUBROUTINE samplexy_closest_init(id)

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
      CALL ef_set_num_work_arrays(id, 2)

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

      SUBROUTINE samplexy_closest_result_limits(id)

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
      SUBROUTINE samplexy_closest_work_size(id)

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

*  Allocate double the length of the axes for REAL*8 work arrays.

      mxl = arg_lo_ss(X_AXIS,ARG1)
      myl = arg_lo_ss(Y_AXIS,ARG1)

      mxh = mxl + 
     .     2* (arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1)
      myh = myl + 
     .     2* (arg_hi_ss(Y_AXIS,ARG1)  - arg_lo_ss(Y_AXIS,ARG1) + 1)

*  xaxdat
      CALL ef_set_work_array_dims_6d(id, 1,
     .                              mxl, 1, 1, 1, 1, 1,
     .                              mxh, 1, 1, 1, 1, 1)

*  yaxdat
      CALL ef_set_work_array_dims_6d(id, 2,
     .                              myl, 1, 1, 1, 1, 1,
     .                              myh, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
*
* In this subroutine we compute the result
*
      SUBROUTINE samplexy_closest_compute(id, arg_1, arg_2, arg_3,
     .        result, xaxdat, yaxdat)

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

      REAL*8 xaxdat(wrk1lox:wrk1lox+(wrk1hix-wrk1lox)/2,wrk1loy:wrk1hiy,
     .              wrk1loz:wrk1hiz, wrk1lot:wrk1hit,
     .              wrk1loe:wrk1hie, wrk1lof:wrk1hif)
      REAL*8 yaxdat(wrk2lox:wrk2lox+(wrk2hix-wrk2lox)/2,wrk2loy:wrk2hiy,
     .              wrk2loz:wrk2hiz, wrk2lot:wrk2hit,
     .              wrk2loe:wrk2hie, wrk2lof:wrk2hif)

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

      LOGICAL TM_FPEQ

      INTEGER nx, nxx, nxy, nxz, nxt, nxe, nxf
      INTEGER ny, nyx, nyy, nyz, nyt, nye, nyf
      INTEGER ndimx, ndimy
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2
      INTEGER i3, j3, k3, l3, m3, n3
      INTEGER imatch, jmatch
      INTEGER ibot, jbot
      REAL xbot, xtop, ybot, ytop
      REAL x_to_xbot, x_to_xtop, y_to_ybot, y_to_ytop
      CHARACTER*255 err_msg

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), modulo(6), regular(6)
      INTEGER xlo, xhi, ylo, yhi
      REAL dx, dy, delmodx, delmody, xpt, ypt

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
      DO 10 i = X_AXIS,F_AXIS
         IF ( arg_hi_ss(i,ARG2) .GT. arg_lo_ss(i,ARG2) )
     .      ndimx = ndimx + 1
         IF ( arg_hi_ss(i,ARG3) .GT. arg_lo_ss(i,ARG3) ) 
     .      ndimy = ndimy + 1
   10 CONTINUE

      IF ( (nx .NE. ny) .OR. (ndimx .GT. 1) .OR. (ndimy .GT. 1) ) THEN
         err_msg =
     .  'Arguments 2 and 3 must be 1-dimensional lists of equal length'
         GO TO 999
      ENDIF

*  Get x and y coordinates of the data to be sampled.

      CALL ef_get_coordinates(id, ARG1, X_AXIS,
     .   arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), xaxdat)
      CALL ef_get_coordinates(id, ARG1, Y_AXIS,
     .   arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), yaxdat)

*  Check to see if input x,y  axis is modulo

      CALL ef_get_axis_info_6d(id, ARG1, ax_name, ax_units,
     .                         backward, modulo, regular)

      xlo = arg_lo_ss(X_AXIS,ARG1)
      xhi = arg_hi_ss(X_AXIS,ARG1)
      ylo = arg_lo_ss(Y_AXIS,ARG1)
      yhi = arg_hi_ss(Y_AXIS,ARG1)

      IF ( modulo(1) ) THEN
         IF (.NOT. regular(1)) THEN
            err_msg = 'cannot handle MODULO axis that is not regular'
            GO TO 999
         ELSE
            dx = xaxdat(xlo+1,1,1,1,1,1) - xaxdat(xlo,1,1,1,1,1)
            delmodx = xaxdat(xhi,1,1,1,1,1) - xaxdat(xlo,1,1,1,1,1) + dx
         ENDIF
      ENDIF
      IF ( modulo(2) ) THEN
         IF (.NOT. regular(2)) THEN
            err_msg = 'cannot handle MODULO axis that is not regular'
            GO TO 999
         ELSE
            dy = yaxdat(ylo+1,1,1,1,1,1) - yaxdat(ylo,1,1,1,1,1)
            delmody = yaxdat(yhi,1,1,1,1,1) - yaxdat(ylo,1,1,1,1,1) + dy
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
            DO 80 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
            DO 80 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
            DO 80 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            DO 80 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
  80        CONTINUE
*           Go to the next coordinates pair
            GOTO 690
         ENDIF

         IF ( modulo(1) ) THEN
* Relocate the X point within the range of modulo X axis 
            DO WHILE ( xpt .GE. xaxdat(xhi,1,1,1,1,1) ) 
               xpt = xpt - delmodx
            ENDDO
            DO WHILE (xpt .LT. xaxdat(xlo,1,1,1,1,1) )
               xpt = xpt + delmodx
            ENDDO
         ENDIF

         ibot = ef_unspecified_int4
         imatch = ef_unspecified_int4

* Check if xpt exceeds the X-axis range.
         i1 = arg_hi_ss(X_AXIS,ARG1)
         IF ( TM_FPEQ(xaxdat(i1,1,1,1,1,1), xpt) ) THEN
            imatch = i1
            GOTO 120
         ELSE IF ( xpt .GT. xaxdat(i1,1,1,1,1,1) ) THEN
            GOTO 120
         ENDIF

* Find xpt in the X-axis range.
         DO 110 i1 = arg_lo_ss(X_AXIS,ARG1), arg_hi_ss(X_AXIS,ARG1)
            IF ( TM_FPEQ(xaxdat(i1,1,1,1,1,1), xpt) ) THEN
               imatch = i1
               GOTO 120
            ELSE IF ( xpt .GT. xaxdat(i1,1,1,1,1,1) ) THEN
               ibot = i1
            ENDIF
  110    CONTINUE

  120    IF ( imatch .NE. ef_unspecified_int4 ) THEN
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

         IF ( modulo(2) ) THEN
* Relocate the Y point within the range of modulo Y axis 
            DO WHILE ( ypt .GE. yaxdat(yhi,1,1,1,1,1) ) 
               ypt = ypt - delmody
            ENDDO
            DO WHILE (ypt .LT. yaxdat(ylo,1,1,1,1,1) )
               ypt = ypt + delmody
            ENDDO
         ENDIF

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
         DO 210 j1 = arg_lo_ss(Y_AXIS,ARG1), arg_hi_ss(Y_AXIS,ARG1)
            IF ( TM_FPEQ(yaxdat(j1,1,1,1,1,1), ypt) ) THEN
               jmatch = j1
               GOTO 220
            ELSE IF ( ypt .GT. yaxdat(j1,1,1,1,1,1) ) THEN
               jbot = j1
            ENDIF
  210    CONTINUE

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
