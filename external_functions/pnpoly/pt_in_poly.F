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
* This function implements the pnpoly code from W. Randolph Franklin found
* in his web pages. The exact code from these web pages is preserved at the end of
* this file; changes to it are the responsibility of Ferret program developers.

* Copyright (c) 1970-2003, Wm. Randolph Franklin
*
* Permission is hereby granted, free of charge, to any person obtaining a 
* copy of this software and associated documentation files (the "Software"), 
* to deal in the Software without restriction, including without limitation 
* the rights to use, copy, modify, merge, publish, distribute, sublicense, 
* and/or sell copies of the Software, and to permit persons to whom the 
* Software is furnished to do so, subject to the following conditions:

*  1. Redistributions of source code must retain the above copyright notice, 
* this list of conditions and the following disclaimers.
*   2. Redistributions in binary form must reproduce the above copyright 
* notice in the documentation and/or other materials provided with the 
* distribution.
*   3. The name of W. Randolph Franklin may not be used to endorse or 
* promote products derived from this Software without specific prior 
* written permission. 
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
* IN THE SOFTWARE. 
*
C W. Randolph Franklin
C http://www.ecse.rpi.edu/Homepages/wrf/pmwiki/
C http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html#The%20C%20Code
C http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html#Fortran%20Code%20for%20the%20Point%20in%20Polygon%20Test

C  RETURNS:
C           -1 IF THE POINT IS OUTSIDE OF THE POLYGON, 
C            0 IF THE POINT IS ON AN EDGE OR AT A VERTEX, 
C            1 IF THE POINT IS INSIDE OF THE POLYGON. 

***********************************************************************
*  Initialize the function

      SUBROUTINE pt_in_poly_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT 
     . ('Return -1 outside, 0 if on edge, 1 if inside polygon')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_num_work_arrays(id, 4)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_desc(id, arg, 
     .  'Variable on the XY grid and region to be tested')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'XVERT')
      CALL ef_set_arg_desc(id, arg, 
     .   'X-coordinates of vertices of polygon')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'YVERT')
      CALL ef_set_arg_desc(id, arg, 
     .   'Y-coordinates of vertices of polygon')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      RETURN 
      END

***********************************************************************

* Request an amount of storage to be supplied by Ferret and passed
* as additional arguments.

      SUBROUTINE pt_in_poly_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

*
* Set the work array X/Y/Z/T dimensions

      INTEGER array_num, nx, ny, nv
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

* These are going to be a double precision array, to contain axis 
* coordinates so allocate 2* the size of the axes

      nx = 2*( arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1 )
      ny = 2*( arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1) + 1 )

      array_num = 1
      CALL ef_set_work_array_dims(id, array_num, 1,1,1,1, nx,1,1,1)
      array_num = 2
      CALL ef_set_work_array_dims(id, array_num, 1,1,1,1, ny,1,1,1)

* Set two arrays the size of the number of vertices.

      nv = arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1 
      nv = MAX(nv, arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1 )
      nv = MAX(nv, arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2) + 1 )
      nv = MAX(nv, arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1 )
      array_num = 3
      CALL ef_set_work_array_dims(id, array_num, 1,1,1,1, nv,1,1,1)
      array_num = 4
      CALL ef_set_work_array_dims(id, array_num, 1,1,1,1, nv,1,1,1)

      RETURN 
      END

* **********************************************************************

      SUBROUTINE pt_in_poly_compute (id, arg_1, arg_2, arg_3, result, 
     .   xcoords, ycoords, workx, worky)
C 
C REMARKS 
C   THE VERTICES MAY BE LISTED CLOCKWISE OR ANTICLOCKWISE. 
C   THE FIRST MAY OPTIONALLY BE REPEATED, IF SO N MAY 
C   OPTIONALLY BE INCREASED BY 1. 
C   THE INPUT POLYGON MAY BE A COMPOUND POLYGON CONSISTING 
C   OF SEVERAL SEPARATE SUBPOLYGONS. IF SO, THE FIRST VERTEX 
C   OF EACH SUBPOLYGON MUST BE REPEATED, AND WHEN CALCULATING 
C   N, THESE FIRST VERTICES MUST BE COUNTED TWICE. 
C   INOUT IS THE ONLY PARAMETER WHOSE VALUE IS CHANGED. 
C   THE SIZE OF THE ARRAYS MUST BE INCREASED IF N > MAXDIM 
C   WRITTEN BY RANDOLPH FRANKLIN, UNIVERSITY OF OTTAWA, 7/70. 
C   
C SUBROUTINES AND FUNCTION SUBPROGRAMS REQUIRED 
C   NONE 
C 
C METHOD 
C   A VERTICAL LINE IS DRAWN THRU THE POINT IN QUESTION. IF IT 
C   CROSSES THE POLYGON AN ODD NUMBER OF TIMES, THEN THE 
C   POINT IS INSIDE OF THE POLYGON. 
C 
C .................................................................. 
C 
      

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, 
     .           mem2loz:mem2hiz, mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, 
     .           mem3loz:mem3hiz, mem3lot:mem3hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit)
      REAL*8 xcoords(wrk1lox:wrk1hix)
      REAL*8 ycoords(wrk2lox:wrk2hix)
      REAL workx(wrk3lox:wrk3hix)
      REAL worky(wrk4lox:wrk4hix)

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      INTEGER n, i,j,k,l, i1,j1, ii,jj, iarg
      LOGICAL MX,MY,NX,NY 

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* Get the x and y coordinates of the input variable

      iarg = 1
      CALL ef_get_coordinates (id, iarg, X_AXIS, 
     .       arg_lo_ss(X_AXIS, ARG1), arg_hi_ss(X_AXIS, ARG1), xcoords)
      CALL ef_get_coordinates (id, iarg, Y_AXIS, 
     .       arg_lo_ss(Y_AXIS, ARG1), arg_hi_ss(Y_AXIS, ARG1), ycoords)

! Size of the list of polygon vertices

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

* Loop over all points, checking if they are inside or outside the polygon.

      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 200 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         PX = xcoords(i1) 

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 100 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            PY = ycoords(j1)

            IF (px .GE. xvmin .AND. px .LE. xvmax .AND.
     .          py .GE. yvmin .AND. py .LE. yvmax ) THEN
               CALL PNPOLY (px, py, arg_2, arg_3, nv, 
     .                      workx, worky, inout )
               result(i,j,k,l)= FLOAT(inout)
            ELSE
               result(i,j,k,l) = -1
            ENDIF

      j1 = j1 + arg_incr(Y_AXIS,ARG1)
 100  CONTINUE

      i1 = i1 + arg_incr(X_AXIS,ARG1)
 200  CONTINUE

      RETURN 
      END



