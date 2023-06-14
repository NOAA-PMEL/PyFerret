*
* pt_in_poly.F
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

** Ansley Manke
* January 2008
*
C 11/2022 replace the pnpoly function with an f90 code which is used without any
C         revisions.
C
C  RETURNS:
C           -1 IF THE POINT IS OUTSIDE OF THE POLYGON, 
C            0 IF THE POINT IS ON AN EDGE OR AT A VERTEX, 
C            1 IF THE POINT IS INSIDE OF THE POLYGON. 

***********************************************************************
*  Initialize the function

      SUBROUTINE pt_in_poly_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CHARACTER*100 fcn_desc

      fcn_desc =
     . 'Return -1 outside, 0 if on edge, 1 if inside polygon'
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                NORMAL,          NORMAL,
     .                                NORMAL,          NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 4)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_desc(id, arg, 
     .  'Variable on the XY grid and region to be tested')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'XVERT')
      CALL ef_set_arg_desc(id, arg, 
     .   'X-coordinates of vertices of polygon')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'YVERT')
      CALL ef_set_arg_desc(id, arg, 
     .   'Y-coordinates of vertices of polygon')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      RETURN 
      END

***********************************************************************

* Request an amount of storage to be supplied by Ferret and passed
* as additional arguments.

      SUBROUTINE pt_in_poly_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

*
* Set the work array X/Y/Z/T/E/F dimensions
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)

      INTEGER array_num, nx, ny, nv
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

* Work arrays contain the coordinate data from argument 1

      array_num = 1
      CALL ef_set_work_array_dims_6d(id, array_num,
     .                               arg_lo_ss(X_AXIS,ARG1), 1, 1, 1, 1, 1,
     .                               arg_hi_ss(X_AXIS,ARG1), 1, 1, 1, 1, 1)

      array_num = 2
      CALL ef_set_work_array_dims_6d(id, array_num,
     .                               arg_lo_ss(Y_AXIS,ARG1), 1, 1, 1, 1, 1,
     .                               arg_hi_ss(Y_AXIS,ARG1), 1, 1, 1, 1, 1)

      RETURN 
      END

* **********************************************************************

      SUBROUTINE pt_in_poly_compute (id, arg_1, arg_2, arg_3, result, 
     .   xcoords, ycoords)

C 
C Originally written as an external function, this version uses
C internal Ferret functions STR_UPCASE and TM_HAS_STRING to upcase
C the axis units string and compare its value to legal longitude
C units strings.
C
C .................................................................. 
C 

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

      REAL*8 xcoords(wrk1lox:wrk1hix)
      REAL*8 ycoords(wrk2lox:wrk2hix)

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
      
      CHARACTER ax_name(6)*16, ax_units(6)*16, dummy*16
      LOGICAL backward(6), mdulo(6), regular(6)

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, ii, jj, iarg, inout, nv
      REAL xvmin, xvmax, yvmin, yvmax, px, py, xmin, xmax
      INTEGER STR_UPCASE
      LOGICAL TM_HAS_STRING, mod_lo, mod_hi, its_dsg
      
      CHARACTER*128 err_msg
      
* Does the variable come from a Discrete Sampling Geometries dataset?  
* If so the function doesn't apply.

      CALL EF_GET_ITS_DSG(id, ARG1, its_dsg )
      IF (its_dsg) THEN
         err_msg = 
     . 'Data is from a Discrete Sampling Geometries (DSG) dataset. Function requires an XY grid.'
	GOTO 999
      ENDIF

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* Get the x and y coordinates of the input variable

      iarg = 1
      CALL ef_get_coordinates (id, iarg, X_AXIS, 
     .       arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), xcoords)
      CALL ef_get_coordinates (id, iarg, Y_AXIS, 
     .       arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), ycoords)

* Size of the list of polygon vertices

      nv = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1 
      nv = MAX(nv, arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1 )
      nv = MAX(nv, arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2) + 1 )
      nv = MAX(nv, arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1 )


* Get the bounding box around the polygon. Do not check points if they are
* outside this bounding box
 
      xvmin = ABS( bad_flag(ARG3) )
      xvmax = -1*xvmin
      yvmin = ABS( bad_flag(ARG4) )
      yvmax = -1*yvmin

      CALL POLY_BOUND_BOX(arg_2, arg_3, nv, xvmin, xvmax, yvmin, yvmax)

* If x axis of arg 1 is longitude, see if the xpts are outside the range of the x axis. Do we need to work 
* in a different modulo branch?
* Get the units of the x axis of arg1

      CALL ef_get_axis_info_6d(id, ARG1, ax_name, ax_units,
     .                         backward, mdulo, regular)
      dummy = ax_units(1) 
      i = STR_UPCASE (dummy, ax_units(1))
      
      mod_lo = .FALSE.
      mod_hi = .FALSE.

      IF ( TM_HAS_STRING(ax_units(1), 'EAST')
     .   .OR. TM_HAS_STRING(ax_units(1), 'WEST')
     .   .OR. TM_HAS_STRING(ax_units(1), 'degreeE')
     .   .OR. TM_HAS_STRING(ax_units(1), 'degreesE')
     .   .OR. TM_HAS_STRING(ax_units(1), 'degree_E')
     .   .OR. TM_HAS_STRING(ax_units(1), 'degrees_E')
     .   .OR. TM_HAS_STRING(ax_units(1), 'degreeW')
     .   .OR. TM_HAS_STRING(ax_units(1), 'degreesW')
     .   .OR. TM_HAS_STRING(ax_units(1), 'degree_W')
     .   .OR. TM_HAS_STRING(ax_units(1), 'degrees_W')
     .   .OR. TM_HAS_STRING(ax_units(1), 'LON') ) THEN


         xmin = 1.e34
         xmax = -1.e34
         do i = wrk1lox,wrk1hix 
            xmin = MIN(xmin, xcoords(i))
            xmax = MAX(xmax, xcoords(i))
         enddo

         mod_lo = (xvmin .LT. xmin)
         mod_hi = (xvmax .GT. xmax)
      ENDIF

* Loop over all points, checking if they are inside or outside the polygon.

      n = res_lo_ss(F_AXIS)
      m = res_lo_ss(E_AXIS)
      l = res_lo_ss(T_AXIS)
      k = res_lo_ss(Z_AXIS)

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
         py = ycoords(j1)

         i1 = arg_lo_ss(X_AXIS,ARG1)
         DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
            px = xcoords(i1) 

            IF (px .GE. xvmin .AND. px .LE. xvmax .AND.
     .          py .GE. yvmin .AND. py .LE. yvmax ) THEN
               CALL PNPOLY (px, py, arg_2, arg_3, nv, 
     .                      inout )
               result(i,j,k,l,m,n)= FLOAT(inout)
            ELSE
               result(i,j,k,l,m,n) = -1
            ENDIF
	    
	    IF (mod_lo) THEN
            IF (px-360 .GE. xvmin .AND. px-360 .LE. xvmax .AND.
     .          py .GE. yvmin .AND. py .LE. yvmax ) THEN
               CALL PNPOLY (px-360, py, arg_2, arg_3, nv, 
     .                      inout )
               result(i,j,k,l,m,n)= FLOAT(inout)
            ELSE
               result(i,j,k,l,m,n) = -1
            ENDIF
	    ENDIF  ! mod_lo

	    IF (mod_hi) THEN
            IF (px+360 .GE. xvmin .AND. px+360 .LE. xvmax .AND.
     .          py .GE. yvmin .AND. py .LE. yvmax ) THEN
               CALL PNPOLY (px+360, py, arg_2, arg_3, nv, 
     .                      inout )
               result(i,j,k,l,m,n)= FLOAT(inout)
            ELSE
               result(i,j,k,l,m,n) = -1
            ENDIF
	    ENDIF  ! mod_hi

            i1 = i1 + arg_incr(X_AXIS,ARG1)
 100     CONTINUE

         j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200  CONTINUE

      RETURN 

 999  CALL ef_bail_out (id, err_msg)

      RETURN 
      END




C .................................................................. 

      SUBROUTINE POLY_BOUND_BOX (xv, yv, nv, xvmin, xvmax, yvmin, yvmax)

* Get the bounding box around the polygon. min and max variables
* have been initialized.

      REAL xv(*), yv(*), xvmin, xvmax, yvmin, yvmax
      INTEGER nv
      INTEGER i

      DO 50 i = 1, nv
         xvmin = MIN(xvmin, xv(i))
         xvmax = MAX(xvmax, xv(i))
         yvmin = MIN(yvmin, yv(i))
         yvmax = MAX(yvmax, yv(i))
  50  CONTINUE

      RETURN
      
      END

!  11/2022 replace pnpoly with this f90 version:
! Source File : PNPOLY.f90 
! https://fortran.uk/HyperKWIC/AERMOD/_kwic/src/236.htm
! 
!----------------------------------------------------------------------
!     Courtesy: Jay Sandhu
!               email: jsandhu@esri.com
!
!
! Please cite David H. Douglas, COLLECTED ALGORITHMS, Cambridge MA:
! Harvard Laboratory for Computer Graphics, 1974
!
! This is my reinvention buster.
! 1974 1974 1974 1974 1974 1974 1974 1974 1974 1974 1974 1974
!
!>>>PNPY
!     .................................................................
!
!        SUBROUTINE PNPOLY
!
!        PURPOSE
!           TO DETERMINE WHETHER A POINT IS INSIDE A POLYGON
!
!        USAGE
!           CALL PNPOLY (PX, PY, X, Y, N, INOUT )
!
!        DESCRIPTION OF THE PARAMETERS
!           PX      - X-COORDINATE OF POINT IN QUESTION.
!           PY      - Y-COORDINATE OF POINT IN QUESTION.
!           X       - N LONG VECTOR CONTAINING X-COORDINATES OF
!                     VERTICES OF POLYGON.
!           Y       - N LONG VECTOR CONTAINING Y-COORDINATES OF
!                     VERTICES OF POLYGON.
!           N       - NUMBER OF VERTICES IN THE POLYGON.
!           INOUT   - THE SIGNAL RETURNED:
!                     -1 IF THE POINT IS OUTSIDE OF THE POLYGON,
!                      0 IF THE POINT IS ON AN EDGE OR AT A VERTEX,
!                      1 IF THE POINT IS INSIDE OF THE POLYGON.
!
!        REMARKS
!           THE VERTICES MAY BE LISTED IN CLOCKWISE OR ANTICLOCKWISE
!           ORDER.  FOR THIS SUBROUTINE A POINT IS CONSIDERED INSIDE
!           THE POLYGON IF IT IS LOCATED IN THE ENCLOSED AREA DEFINED
!           BY THE LINE FORMING THE POLYGON.
!           THE INPUT POLYGON MAY BE A COMPOUND POLYGON CONSISTING
!           OF SEVERAL SEPARATE SUBPOLYGONS. IF SO, THE FIRST VERTEX
!           OF EACH SUBPOLYGON MUST BE REPEATED, AND WHEN CALCULATING
!           N, THESE FIRST VERTICES MUST BE COUNTED TWICE.
!           INOUT IS THE ONLY PARAMETER WHOSE VALUE IS CHANGED.
!           PNPOLY CAN HANDLE ANY NUMBER OF VERTICES IN THE POLYGON.
!           WRITTEN BY RANDOLPH FRANKLIN, UNIVERSITY OF OTTAWA, 6/72.
!
!        SUBROUTINES AND FUNCTION SUBPROGRAMS REQUIRED
!           NONE
!
!        METHOD
!           A VERTICAL SEMI-INFINITE LINE IS DRAWN UP FROM THE POINT
!           IN QUESTION. IF IT CROSSES THE POLYGON AN ODD NUMBER OF
!           TIMES, THE POINT IS INSIDE THE POLYGON.
!
!     .................................................................
!
      SUBROUTINE PNPOLY(PX,PY,X,Y,N,INOUT)

      IMPLICIT NONE

      INTEGER I , J , N , INOUT
      REAL X(N) , Y(N) , XI , YI , XJ , YJ , PX , PY
      LOGICAL IX , IY , JX , JY , EOR

!     EXCLUSIVE OR STATEMENT FUNCTION.
      EOR(IX,IY) = (IX .OR. IY) .AND. .NOT.(IX .AND. IY)

      INOUT = -1                                               

      DO I = 1 , N
         XI = X(I) - PX                                        
         YI = Y(I) - PY
!        CHECK WHETHER THE POINT IN QUESTION IS AT THIS VERTEX.
         IF ( XI.EQ.0.0 .AND. YI.EQ.0.0 ) THEN
            INOUT = 0                                          
            RETURN
         ENDIF
!        J IS NEXT VERTEX NUMBER OF POLYGON.
         J = 1 + MOD(I,N)                                      
         XJ = X(J) - PX
         YJ = Y(J) - PY
!        IS THIS LINE OF 0 LENGTH ?
         IF ( XI.EQ.XJ .AND. YI.EQ.YJ ) GOTO 100
         IX = XI.GE.0.0                                        
         IY = YI.GE.0.0
         JX = XJ.GE.0.0
         JY = YJ.GE.0.0
!        CHECK WHETHER (PX,PY) IS ON VERTICAL SIDE OF POLYGON.
         IF ( XI.EQ.0.0 .AND. XJ.EQ.0.0 .AND. EOR(IY,JY) ) THEN
            INOUT = 0                                          
            RETURN
         ENDIF
!        CHECK WHETHER (PX,PY) IS ON HORIZONTAL SIDE OF POLYGON.
         IF ( YI.EQ.0.0 .AND. YJ.EQ.0.0 .AND. EOR(IX,JX) ) THEN
            INOUT = 0                                          
            RETURN
         ENDIF
!        CHECK WHETHER BOTH ENDS OF THIS SIDE ARE COMPLETELY 1) TO RIGHT
!        OF, 2) TO LEFT OF, OR 3) BELOW (PX,PY).
         IF ( .NOT.((IY .OR. JY) .AND. EOR(IX,JX)) ) GOTO 100  
!        DOES THIS SIDE OBVIOUSLY CROSS LINE RISING VERTICALLY FROM (PX,PY)
         IF ( .NOT.(IY .AND. JY .AND. EOR(IX,JX)) ) THEN       
            IF ( (YI*XJ-XI*YJ)/(XJ-XI).LT.0.0 ) THEN           
               GOTO 100                                        
            ELSEIF ( (YI*XJ-XI*YJ)/(XJ-XI).EQ.0.0 ) THEN
               INOUT = 0                                       
               RETURN
            ELSE
               INOUT = -INOUT                                  
            ENDIF
         ELSE
            INOUT = -INOUT                                     
         ENDIF

 100  ENDDO

      CONTINUE                                                 
      END
