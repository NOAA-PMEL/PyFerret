*
*  ave_scat2grid_t.F
*
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
* Sep 2004

* Inputs are a set of T coordinates and a variable on an abstract axis, and an 
* output time axis. Returns a time series which is the average of the variable, 
* based on the number of points that fall inside of each T cell. 
* From scat2grid_t.F

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


      SUBROUTINE  ave_scat2grid_t_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Compute average of a variable in each cell of output time axis')
      CALL ef_set_num_args(id, 3)
      CALL ef_set_axis_inheritance_6d(id,
     .                                NORMAL, NORMAL,
     .                                NORMAL, IMPLIED_BY_ARGS,
     .                                NORMAL, NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 3)

* output grid is determined by argument 3. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg, 
     .        'T coordinates of scattered input time coordinates')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'VPTS')
      CALL ef_set_arg_desc(id, arg, 
     .        'Variable at times in TPTS')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'TAXIS')
      CALL ef_set_arg_desc(id, arg, 
     .        'Output time axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, YES, NO, NO)

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
      SUBROUTINE ave_scat2grid_t_work_size(id)

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
*                           xlo, ylo, zlo, tlo, elo, flo
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER ntout, nt2

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      ntout = 1 + arg_hi_ss(T_AXIS,ARG3) - arg_lo_ss(T_AXIS,ARG3)

* doubled in case REAL is REAL*4
      nt2 = 2 * ntout

* taxlo  output t axis box lower limits
      CALL ef_set_work_array_dims_6d(id, 1,
     .                                1, 1, 1, 1, 1, 1,
     .                              nt2, 1, 1, 1, 1, 1)

* taxhi  output t axis box upper limits
      CALL ef_set_work_array_dims_6d(id, 2,
     .                                1, 1, 1, 1, 1, 1,
     .                              nt2, 1, 1, 1, 1, 1)

* counts 
      CALL ef_set_work_array_dims_6d(id, 3, 
     .                                1, 1, 1, 1, 1, 1,
     .                            ntout, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*

      SUBROUTINE ave_scat2grid_t_compute(id, arg_1, arg_2, arg_3,
     .                               result, taxlo, taxhi, counts)

*  arg_1  tpts  Scattered t values to be gridded.
*  arg_2  vpts  Scattered fcn values to be gridded.
*  arg_3  taxis to use for output

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

* Ignore the unused final dimensions in the work arrays since
* Fortran is column major
      REAL*8 taxlo(wrk1lox:wrk1hix/2)
      REAL*8 taxhi(wrk2lox:wrk2hix/2)
      REAL counts(wrk3lox:wrk3hix)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

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
      INTEGER idx, lo_ss, hi_ss, l1
      INTEGER ntpts, nvpts, ntgrid
      REAL tt, vv
      CHARACTER*250 errtxt

*  variables for checking axis characteristics (modulo axes)
      CHARACTER ax_name(6)*16, ax_units(6)*16
      LOGICAL backward(6), mdulo(6), regular(6)
      LOGICAL modulot
      REAL*8 modtlen

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  Check to see if output axis is modulo (does not need to be regular)
      CALL ef_get_axis_info_6d(id, ARG3, ax_name, ax_units,
     .                         backward, mdulo, regular)
      modulot = mdulo(T_AXIS)
      IF ( modulot ) THEN
         CALL ef_get_axis_modulo_len(id, ARG3, T_AXIS, modtlen)
      ENDIF

*  Find number of points in scattered input points.  1-D arrays defining
*  the scattered data points may lie on any axis of the input argument.
*  Using as flattened arrays, so consider all axes.

      ntpts = 0
      DO 40 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG1)
         hi_ss = arg_hi_ss(idx,ARG1)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( ntpts .NE. 0 ) THEN
               ntpts = ntpts * (hi_ss - lo_ss + 1)
            ELSE
               ntpts = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  40  CONTINUE

* Check that arguments 1 and 2 have the same number of values

      nvpts = 0
      DO 80 idx = X_AXIS, F_AXIS
         lo_ss = arg_lo_ss(idx,ARG2)
         hi_ss = arg_hi_ss(idx,ARG2)
         IF ( lo_ss .NE. ef_unspecified_int4 ) THEN
            IF ( nvpts .NE. 0 ) THEN
               nvpts = nvpts * (hi_ss - lo_ss + 1)
            ELSE
               nvpts = hi_ss - lo_ss + 1
            ENDIF
         ENDIF
  80  CONTINUE

      IF ( nvpts .NE. ntpts ) THEN
         errtxt = 'TPTS and VPTS have different number of values'
         GOTO 999
      ENDIF
      IF ( nvpts .EQ. 0 ) THEN
         errtxt = 'No data in TPTS and VPTS'
         GOTO 999
      ENDIF

*  Check tax is a T axis

      lo_ss = arg_lo_ss(T_AXIS,ARG3)
      hi_ss = arg_hi_ss(T_AXIS,ARG3)
*  Check that tax a T axis
      IF ( lo_ss .EQ. ef_unspecified_int4 ) THEN
         errtxt = 'Third argument must be a T axis'
         GO TO 999
      ENDIF
      ntgrid = hi_ss - lo_ss + 1

*  Get cell boundaries

      CALL ef_get_box_limits(id, ARG3, T_AXIS, arg_lo_ss(T_AXIS, ARG3),
     .                       arg_hi_ss(T_AXIS, ARG3), taxlo, taxhi)

*  Compute result

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      m = res_lo_ss(E_AXIS)
      n = res_lo_ss(F_AXIS)

*  Initialize result

      DO l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         result(i,j,k,l,m,n) = 0.
      ENDDO

*  Initialize counts

      DO l1 = 1, ntgrid
         counts(l1) = 0.
      ENDDO

*  Loop over input tpts

      DO 700 idx = 1, ntpts

         CALL pickout2(arg_1, arg_2, idx, tt, vv)
         IF ( tt .EQ. bad_flag(ARG1) .OR. vv .EQ. bad_flag(ARG2) ) THEN
            GOTO 700
         ENDIF

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.
         IF ( modulot ) THEN
            CALL modscat(taxlo(1), taxhi(ntgrid), modtlen, 1, tt)
         ENDIF

         l1 = 1
         DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

            IF ( tt .GE. taxlo(l1) .AND. tt .LE. taxhi(l1) ) THEN
               result(i,j,k,l,m,n) = result(i,j,k,l,m,n) + vv
               counts(l1) = counts(l1) + 1.0
*              Location found and point assigned; go to the next scatter point
               GOTO 700
            ENDIF

            l1 = l1 + 1
 400     CONTINUE

 700  CONTINUE

* Compute average

      l1 = 1
      DO 410 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         IF ( counts(l1) .GT. 0.0 ) THEN
           result(i,j,k,l,m,n) = result(i,j,k,l,m,n) / counts(l1)
         ELSE
           result(i,j,k,l,m,n) = bad_flag_result
         ENDIF

         l1 = l1 + 1
 410  CONTINUE

      RETURN

 999  CALL EF_BAIL_OUT(id, errtxt)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      RETURN
      END
