*
* zaxreplace_rev.F
! version of zaxreplace_avg, with order of arguments 1 and 2 reversed
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
* Oct 1998
* 11/2001  clarify function description
*  7/2003  Bug fix; last source point wasnt associated with a dest box
*          when the destination axis continues beyond the source range
*
*
* Replace Z axis, using weighted averaging over bins
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


      SUBROUTINE zaxreplace_rev_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT('regrid V onto Z axis of ZAX based on Z values in ZVALS ',
     .       'using weighted average')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 7)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'ZVALS')
      CALL ef_set_arg_desc(id, arg,
     .        'Destination Z axis values as a fcn of source Z axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'V')
      CALL ef_set_arg_desc(id, arg,
     .        'Variable on native Z axis')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, NO, YES, YES, YES)

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
      SUBROUTINE zaxreplace_rev_work_size(id)

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
      INTEGER mz1, mz3
      INTEGER iwork

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

*  Allocate double the dimension of the input arguments for work arrays
*  which will be REAL*8
      mz1 = 1 + ABS(arg_hi_ss(Z_AXIS,ARG1) - arg_lo_ss(Z_AXIS,ARG1))
      mz1 = 2 * mz1
      mz3 = 1 + ABS(arg_hi_ss(Z_AXIS,ARG3) - arg_lo_ss(Z_AXIS,ARG3))
      mz3 = 2 * mz3

*  zaxsrc
      iwork = 1
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                              mz1, 1, 1, 1, 1, 1)

*  sbox_lo_lim
      iwork = 2
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                              mz1, 1, 1, 1, 1, 1)

*  sbox_hi_lim
      iwork = 3
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                              mz1, 1, 1, 1, 1, 1)

*  sbox_lo_new
      iwork = 4
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                              mz1, 1, 1, 1, 1, 1)

*  sbox_hi_new
      iwork = 5
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                              mz1, 1, 1, 1, 1, 1)

*  dbox_lo_lim
      iwork = 6
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                              mz3, 1, 1, 1, 1, 1)

*  dbox_hi_lim
      iwork = 7
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                              mz3, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE zaxreplace_rev_compute(id, arg_1, arg_2, arg_3,
     .                                  result, zaxsrc,
     .                                  sbox_lo_lim, sbox_hi_lim,
     .                                  sbox_lo_new, sbox_hi_new,
     .                                  dbox_lo_lim, dbox_hi_lim)

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
      REAL*8 zaxsrc     (wrk1lox:wrk1hix/2)
      REAL*8 sbox_lo_lim(wrk2lox:wrk2hix/2)
      REAL*8 sbox_hi_lim(wrk3lox:wrk3hix/2)
      REAL*8 sbox_lo_new(wrk4lox:wrk4hix/2)
      REAL*8 sbox_hi_new(wrk5lox:wrk5hix/2)
      REAL*8 dbox_lo_lim(wrk6lox:wrk6hix/2)
      REAL*8 dbox_hi_lim(wrk7lox:wrk7hix/2)

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
      INTEGER nbox, nsrc
      INTEGER nsrc_lo, nsrc_hi
      REAL frac, xsrc_min, xsrc_max, xsrc_lo, xsrc_hi, xsrcdiff
      REAL wtd, arg2_val, arg2_val_last, wtd_last
      REAL*8 sum, wtdsum
      REAL dbox_lo, dbox_hi

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      CALL ef_get_coordinates(id, ARG1, Z_AXIS,
     .        arg_lo_ss(Z_AXIS, ARG1), arg_hi_ss(Z_AXIS, ARG1), zaxsrc)

      CALL ef_get_box_limits(id, ARG1, Z_AXIS, arg_lo_ss(Z_AXIS, ARG1),
     .        arg_hi_ss(Z_AXIS, ARG1), sbox_lo_lim, sbox_hi_lim)

      CALL ef_get_box_limits(id, ARG3, Z_AXIS, arg_lo_ss(Z_AXIS, ARG3),
     .        arg_hi_ss(Z_AXIS, ARG3), dbox_lo_lim, dbox_hi_lim)

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

*
* -ACM- Note: need to think about backwards and forwards z axes: ht vs depth ?
*

*  Translate the source boxes to the new axis units.

         DO 310 k = arg_lo_ss(Z_AXIS,ARG2), arg_hi_ss(Z_AXIS,ARG2)

*  Find the point on the source axis just below the box limit.  Interpolate
*  to get the box limit in new units.
            nbox = 1
            DO WHILE ( zaxsrc(nbox) .LT. sbox_lo_lim(k) )
               nbox = nbox + 1
            ENDDO
            IF ( nbox .EQ. 1 ) THEN
               sbox_lo_new(k) = arg_1(i1,j1,nbox,l1,m1,n1)
            ELSE
               frac = ( sbox_lo_lim(k) - zaxsrc(nbox-1) ) /
     .                ( zaxsrc(nbox)   - zaxsrc(nbox-1) )
               sbox_lo_new(k) = arg_1(i1,j1,nbox-1,l1,m1,n1) +
     .                          frac * ( arg_1(i1,j1,nbox,  l1,m1,n1) -
     .                                   arg_1(i1,j1,nbox-1,l1,m1,n1) )
            ENDIF

            nbox = 1
            DO WHILE ( zaxsrc(nbox) .LT. sbox_hi_lim(k) .AND.
     .                 nbox .LT. arg_hi_ss(Z_AXIS,ARG2) )
               nbox = nbox + 1
            ENDDO
            IF ( nbox .EQ. 1 ) THEN
               sbox_hi_new(k) = arg_1(i1,j1,nbox,l1,m1,n1)
            ELSE
               frac = ( sbox_hi_lim(k) - zaxsrc(nbox-1) ) /
     .                ( zaxsrc(nbox)   - zaxsrc(nbox-1) )
               sbox_hi_new(k) = arg_1(i1,j1,nbox-1,l1,m1,n1) +
     .                          frac * ( arg_1(i1,j1,nbox,  l1,m1,n1) -
     .                                   arg_1(i1,j1,nbox-1,l1,m1,n1) )
            ENDIF

  310    CONTINUE

* For each z line accumulate weighted source points and weights to get
* the average value.   Loop through the each destination grid box.

         sum = 0.
         wtdsum = 0.
         DO 390 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*  Find the first and last source point in this destination box
            nsrc_lo = 1
            DO WHILE ( sbox_lo_new(nsrc_lo) .LT. dbox_lo_lim(k) )
               nsrc_lo = nsrc_lo + 1
               IF ( nsrc_lo .GT. arg_hi_ss(Z_AXIS, ARG1) ) GO TO 320
            ENDDO
  320       CONTINUE

*  Find any portion of the source box lying in the destination box.
            nsrc_hi = 1
            DO WHILE ( sbox_lo_new(nsrc_hi) .LE. dbox_hi_lim(k) )
               nsrc_hi = nsrc_hi + 1
               IF ( nsrc_hi .GT. arg_hi_ss(Z_AXIS, ARG1) ) GO TO 330
            ENDDO
  330       CONTINUE
            nsrc_hi = nsrc_hi - 1

            IF ( nsrc_hi .LT. nsrc_lo ) THEN
*  No source boxes within this destination box.
               IF ( wtdsum .NE. 0. ) THEN
                  result(i,j,k,l,m,n) = sum / wtdsum
                  sum = 0.
                  wtdsum = 0.
               ELSE
                  result(i,j,k,l,m,n) = bad_flag_result
               ENDIF
*  Done with this Z result; go to the next one
               GOTO 390
            ENDIF

            DO 340 nsrc = nsrc_lo, nsrc_hi

*  Get fraction of this source box lying within the destination box.
               xsrc_min = sbox_lo_new(nsrc)
               xsrc_max = sbox_hi_new(nsrc)

               dbox_hi = dbox_hi_lim(k)
               dbox_lo = dbox_lo_lim(k)

               xsrc_lo = MAX(xsrc_min, dbox_lo)
               xsrc_hi = MIN(xsrc_max, dbox_hi)

               xsrcdiff = xsrc_max - xsrc_min
               wtd = 0.
               IF ( xsrcdiff .NE. 0. )
     .            wtd = (xsrc_hi - xsrc_lo) / xsrcdiff
               arg2_val = arg_2(i2,j2,nsrc,l2,m2,n2)
               IF ( arg2_val .NE. bad_flag(ARG2) ) THEN
                  sum  = sum + wtd * arg2_val
                  wtdsum = wtdsum + wtd
                  arg2_val_last = arg2_val
                  wtd_last = wtd
               ENDIF

  340       CONTINUE

            IF ( wtdsum .NE. 0. ) THEN

* If we had any good data, calculate the average.
               result(i,j,k,l,m,n) = sum / wtdsum

*  Start next average with remaining weighted part of the last box
*  which was not within the destination box.
               sum = (1.- wtd_last) * arg2_val_last
               wtdsum = 1. - wtd_last

            ELSE

               result(i,j,k,l,m,n) = bad_flag_result
               sum = 0.
               wtdsum = 0.

            ENDIF

 390     CONTINUE

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
