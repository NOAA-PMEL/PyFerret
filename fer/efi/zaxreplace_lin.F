*
* zaxreplace_lin.F
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
*
* V580:10/04 *acm* Fix for bug 1037. Before the loop calling hunt_r4,
*                  set the initial guess for nsrc_lo to 1 not 0.  
*       4/06 *acm* as an external function this file included hunt_r4
*                  from the Ferret code. Linking this statically, no need
*                  for this any more.
*
* Replace Z axis, using linear interpolation 
* (same as zaxreplace, as first step towards zaxreplace_zlev)
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


      SUBROUTINE zaxreplace_lin_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 descr

      CALL ef_set_desc(id, 
     . 'regrid V onto Z axis of ZAX based on layer thickness ')

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, YES, YES, YES, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'V')
      CALL ef_set_arg_desc(id, arg, 
     .        'Variable on native z axis')
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'THICKNESS')
      WRITE (descr, 20)
   20 FORMAT ('Thickness corresponding to input Z axis values ',
     .        'in units of output Z axis')

      CALL ef_set_arg_desc(id, arg, descr)
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'ZAX')
      CALL ef_set_arg_desc(id, arg, 
     .'Variable with desired z (depth) axis points')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, NO)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE zaxreplace_lin_compute(id, arg_1, arg_2, arg_3, result)

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
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

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

      INTEGER i, j, k, l
      INTEGER i1, j1, l1, i2, j2, l2, i3, j3, l3
      INTEGER nsrc_lo, klo, khi
      REAL frac, zval

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* 
*  Loop over x,y,t of the input fields
*

      klo = arg_lo_ss(Z_AXIS, ARG2)
      khi = arg_hi_ss(Z_AXIS, ARG2)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      i2 = arg_lo_ss(X_AXIS,ARG2)
      i3 = arg_lo_ss(X_AXIS,ARG3)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         j2 = arg_lo_ss(Y_AXIS,ARG2)
         j3 = arg_lo_ss(Y_AXIS,ARG3)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            l1 = arg_lo_ss(T_AXIS,ARG1)
            l2 = arg_lo_ss(T_AXIS,ARG2)
            l3 = arg_lo_ss(T_AXIS,ARG3)
            DO 200 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

* --- first set this to do regular linear interpolation, as in zaxreplace,
*     and check that it replicates the behavior of zaxreplace.  Then
*     deal with layer thicknesses.

* For each z line interpolate. 

               nsrc_lo = 0

               nsrc_lo = 1    ! fixes bug where if zval equals 1st point in axis
                              ! we got no match.

               DO 100 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

* ... next Z axis value (arg3) to locate

                  zval = arg_3(i3,j3,k,l3)

* ... search the Z field (arg_2) for the index just below this value
* ... note that nsrc_lo is always referenced to a starting index of 1

	          CALL HUNT_R4(arg_2, 1, khi, zval, nsrc_lo, frac)

* ... interpolate data arg_1 field to get result

                  IF ( frac .LT. 0.0 ) THEN
* ... no luck -- the Z axis value we seek is outside the range in com2
                     result(i,j,k,l) = bad_flag_result
                  ELSEIF ( arg_1(i1,j1,nsrc_lo,l1) .EQ. 
     .                            bad_flag(ARG1) ) THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSEIF ( frac .EQ. 1.0 ) THEN                  ! exactly on point
	             result(i,j,k,l) = arg_1(i1,j1,nsrc_lo,l1)
                  ELSE
	             result(i,j,k,l) =  frac* arg_1(i1,j1,nsrc_lo,l1)
     .			     + (1-frac)* arg_1(i1,j1,nsrc_lo+1,l1)
               ENDIF
 100        CONTINUE


               l1 = l1 + arg_incr(T_AXIS,ARG1)
               l2 = l2 + arg_incr(T_AXIS,ARG2)
               l3 = l3 + arg_incr(T_AXIS,ARG3)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
            j2 = j2 + arg_incr(Y_AXIS,ARG2)
            j3 = j3 + arg_incr(Y_AXIS,ARG3)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
         i2 = i2 + arg_incr(X_AXIS,ARG2)
         i3 = i3 + arg_incr(X_AXIS,ARG3)
 400  CONTINUE

      
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

