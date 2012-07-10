*
* zaxreplace_zlev.F
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
* august 2004
* V580:10/04 *acm* Fix for bug 1037. Before the loop calling hunt_r4,
*                  set the initial guess for nsrc_lo to 1 not 0.  
*       4/06 *acm* as an external function this file included hunt_r4
*                  from the Ferret code. Linking this statically, no need
*                  for this any more.*
*
*
* Replace Z axis, where input Z axis is layer and 
* second argument is layer thickness
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


      SUBROUTINE zaxreplace_zlev_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*100 descr

      CALL ef_set_desc(id,
     .        'regrid V onto Z axis of ZAX based on layer thickness')
      CALL ef_set_num_args(id, 3)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_num_work_arrays(id, 1)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'V')
      CALL ef_set_arg_desc(id, arg,
     .        'Variable on native Z axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, NO, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'THICKNESS')
      WRITE (descr, 20)
   20 FORMAT ('Thickness corresponding to input Z axis layers ',
     .        'in units of output Z axis')
      CALL ef_set_arg_desc(id, arg, descr)
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'ZAX')
      CALL ef_set_arg_desc(id, arg,
     .        'Variable with desired Z (depth) axis points')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, YES, NO, NO, NO)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN
      END


*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE zaxreplace_zlev_work_size(id)

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
      INTEGER nz
      INTEGER iwork

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nz = 1 + ABS(arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2))

*  zdepth
      iwork = 1
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                               nz, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE zaxreplace_zlev_compute(id, arg_1, arg_2, arg_3,
     .                                   result, zdepth)

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

* Ignore the Y/Z/T/E/F dimensions in the work arrays since
* they are not used and Fortran is column major
      REAL zdepth(wrk1lox:wrk1hix)

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

      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, l1, m1, n1
      INTEGER i2, j2, l2, m2, n2
      INTEGER i3, j3, l3, m3, n3
      INTEGER nsrc_lo, klo, khi
      REAL sum, lasthalf, frac, zval, zmin, zmax

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      klo = arg_lo_ss(Z_AXIS, ARG2)
      khi = arg_hi_ss(Z_AXIS, ARG2)

*
*  Loop over x,y,t,e,f of the input fields
*
      n1 = arg_lo_ss(F_AXIS,ARG1)
      n2 = arg_lo_ss(F_AXIS,ARG2)
      n3 = arg_lo_ss(F_AXIS,ARG3)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m1 = arg_lo_ss(E_AXIS,ARG1)
      m2 = arg_lo_ss(E_AXIS,ARG2)
      m3 = arg_lo_ss(E_AXIS,ARG3)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      l1 = arg_lo_ss(T_AXIS,ARG1)
      l2 = arg_lo_ss(T_AXIS,ARG2)
      l3 = arg_lo_ss(T_AXIS,ARG3)
      DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

      j1 = arg_lo_ss(Y_AXIS,ARG1)
      j2 = arg_lo_ss(Y_AXIS,ARG2)
      j3 = arg_lo_ss(Y_AXIS,ARG3)
      DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      i3 = arg_lo_ss(X_AXIS,ARG3)
      DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

* Compute depths at this location and time. Depth coordinate is at
* half of the current grid cell, plus half of the previous grid cell.
         sum = 0.0
         lasthalf = 0.0
         DO 310 k = klo, khi
            IF ( arg_2(i2,j2,k,l2,m2,n2) .NE. bad_flag(ARG2) ) THEN
               sum = sum + lasthalf + 0.5 * arg_2(i2,j2,k,l2,m2,n2)
               lasthalf = 0.5 * arg_2(i2,j2,k,l2,m2,n2)
            ENDIF
* If there are any layer thickness, fill the end of the array with
* the sum of thicknesses up to the deepest.
            zdepth(k) = sum
  310    CONTINUE

         zmin = zdepth(klo)
         zmax = zdepth(khi)

* For each z line interpolate.
*        set to one instead of zero
*        fixes bug where we got no match if zval equals first point in axis
         nsrc_lo = 1

         DO 390 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            IF ( sum .EQ. 0.0 ) THEN

               result(i,j,k,l,m,n) = bad_flag_result

            ELSE

* ... next Z axis value (arg3) to locate
               zval = arg_3(i3,j3,k,l3,m,n3)

* ... search the Z field (arg_2) for the index just below this value
* ... note that nsrc_lo is always referenced to a starting index of 1
               IF ( (zval .LT. zmin) .OR. (zval .GT. zmax) ) THEN
                  frac = -999.0
               ELSE
                  CALL HUNT_R4 (zdepth,1,khi,zval,nsrc_lo,frac)
               ENDIF

* ... interpolate data arg_1 field to get result
               IF ( frac .LT. 0.0 ) THEN

* ... no luck -- the Z axis value we seek is outside the range in com2
                  result(i,j,k,l,m,n) = bad_flag_result

               ELSEIF ( arg_1(i1,j1,nsrc_lo,l1,m1,n1)
     .                              .EQ. bad_flag(ARG1) ) THEN

                  result(i,j,k,l,m,n) = bad_flag_result

               ELSEIF ( frac .EQ. 1.0 ) THEN         ! exactly on point

                  result(i,j,k,l,m,n) = arg_1(i1,j1,nsrc_lo,l1,m1,n1)

               ELSEIF ( arg_1(i1,j1,nsrc_lo+1,l1,m1,n1)
     .                              .EQ. bad_flag(ARG1) ) THEN

                  result(i,j,k,l,m,n) = bad_flag_result

               ELSE

                  result(i,j,k,l,m,n) =
     .                        frac  * arg_1(i1,j1,nsrc_lo,  l1,m1,n1) +
     .                  (1. - frac) * arg_1(i1,j1,nsrc_lo+1,l1,m1,n1)

               ENDIF

            ENDIF  ! IF ( sum .EQ. 0.0 ) THEN
 390        CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
         i3 = i3 + arg_incr(X_AXIS,ARG3)
 100  CONTINUE

         j1 = j1 + arg_incr(Y_AXIS,ARG1)
         j2 = j2 + arg_incr(Y_AXIS,ARG2)
         j3 = j3 + arg_incr(Y_AXIS,ARG3)
 200  CONTINUE

         l1 = l1 + arg_incr(T_AXIS,ARG1)
         l2 = l2 + arg_incr(T_AXIS,ARG2)
         l3 = l3 + arg_incr(T_AXIS,ARG3)
 400  CONTINUE

         m1 = m1 + arg_incr(E_AXIS,ARG1)
         m2 = m2 + arg_incr(E_AXIS,ARG2)
         m3 = m3 + arg_incr(E_AXIS,ARG3)
 500  CONTINUE

         n1 = n1 + arg_incr(F_AXIS,ARG1)
         n2 = n2 + arg_incr(F_AXIS,ARG2)
         n3 = n3 + arg_incr(F_AXIS,ARG3)
 600  CONTINUE

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN
      END
