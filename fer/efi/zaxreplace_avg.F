*
* zaxreplace_avg.F
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
* 12/2007  Bug 1557 fix: the function already worked for second argument 
*          varying in xyt. But gaps not treated correctly: for each k, we 
*          start next k average with remaining weighted part of the last box
*          When missing data encountered, use this remaining part, then zero
*          out the variables that held it.
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


      SUBROUTINE zaxreplace_avg_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT 
     .  ('regrid V onto Z axis of ZAX based on Z values in ZVALS ',
     .    'using weighted average' )
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, YES, YES, YES, NO)

      CALL ef_set_num_work_arrays(id, 8)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'V')
      CALL ef_set_arg_desc(id, arg, 
     .        'Variable on native z axis')
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'ZVALS')
      CALL ef_set_arg_desc(id, arg, 
     .        'Destination Z axis values as a fcn of source Z axis')
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
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE zaxreplace_avg_work_size(id)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

*
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER mz2, mz3

      INTEGER iwork
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

*  Allocate double the dimension of the input arguments for work arrays
*  which will be REAL*8

      mz2 = 1 + ABS(arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2))
      mz3 = 1 + ABS(arg_hi_ss(Z_AXIS,ARG3) - arg_lo_ss(Z_AXIS,ARG3))

      mz2 = 2* mz2
      mz3 = 2* mz3

*  zaxsrc
      iwork = 1
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   1, 1, mz2, 1)

*  sbox_lo_lim
      iwork = 2
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   1, 1, mz2, 1)

*  sbox_hi_lim
      iwork = 3
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   1, 1, mz2, 1)

*  sbox_lo_new
      iwork = 4
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   1, 1, mz2, 1)

*  sbox_hi_new
      iwork = 5
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   1, 1, mz2, 1)

*  dbox_lo_lim
      iwork = 6
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   1, 1, mz3, 1)

*  dbox_hi_lim
      iwork = 7
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   1, 1, mz3, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE zaxreplace_avg_compute(id, arg_1, arg_2, arg_3, 
     .     result, zaxsrc, sbox_lo_lim, sbox_hi_lim, sbox_lo_new, 
     .     sbox_hi_new, dbox_lo_lim, dbox_hi_lim)

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
      INTEGER ifirst, iw, nbox, nsrc
      INTEGER nsrc_lo, nsrc_hi
      REAL frac, xsrc_min, xsrc_max, xsrc_lo, xsrc_hi, xsrcdiff
      REAL wtd, arg1_val, arg1_val_last, wtd_last
      REAL*8 sum, wtdsum
      REAL dbox_lo, dbox_hi
c      CHARACTER*132 err_msg

*  Dimension work arrays

      REAL*8 zaxsrc(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz/2, wrk1lot:wrk1hit)
      REAL*8 sbox_lo_lim(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz/2, wrk2lot:wrk2hit)
      REAL*8 sbox_hi_lim(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz/2, wrk3lot:wrk3hit)
      REAL*8 sbox_lo_new(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz/2, wrk4lot:wrk4hit)
      REAL*8 sbox_hi_new(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .               wrk5loz:wrk5hiz/2, wrk5lot:wrk5hit)
      REAL*8 dbox_lo_lim(wrk6lox:wrk6hix, wrk6loy:wrk6hiy,
     .               wrk6loz:wrk6hiz/2, wrk6lot:wrk6hit)
      REAL*8 dbox_hi_lim(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .               wrk7loz:wrk7hiz/2, wrk7lot:wrk7hit)

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


      CALL ef_get_coordinates(id, ARG2, Z_AXIS, 
     .   arg_lo_ss(Z_AXIS, ARG2), arg_hi_ss(Z_AXIS, ARG2), zaxsrc)

      CALL ef_get_box_limits(id, ARG2, Z_AXIS, arg_lo_ss(Z_AXIS, ARG2), 
     .   arg_hi_ss(Z_AXIS, ARG2), sbox_lo_lim, sbox_hi_lim)


      CALL ef_get_box_limits(id, ARG3, Z_AXIS, arg_lo_ss(Z_AXIS, ARG3), 
     .   arg_hi_ss(Z_AXIS, ARG3), dbox_lo_lim, dbox_hi_lim)

* 
*  Loop over x,y,t of the input fields
*
      ifirst = 1
      iw = 0

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

*

* -ACM- Note: need to think about backwards and forwards z axes: ht vs depth ?


*  Translate the source boxes to the new axis units.

               DO 500 k = arg_lo_ss(Z_AXIS,ARG1), 
     .                    arg_hi_ss(Z_AXIS,ARG1)

*  Find the point on the source axis just below the box limit.  Interpolate
*  to get the box limit in new units.

                  nbox = 1

                  DO WHILE (zaxsrc(1,1,nbox,1) .LT. 
     .                      sbox_lo_lim(1,1,k,1) )
                     nbox = nbox + 1
                  ENDDO

                  IF (nbox .eq. 1) THEN 
                     sbox_lo_new(1,1,k,1) = arg_2(i2,j2,nbox,l2)
                  ELSE
                     frac = (sbox_lo_lim(1,1,k,1) - 
     .                      zaxsrc(1,1,nbox-1,1))/
     .                   (zaxsrc(1,1,nbox,1) - zaxsrc(1,1,nbox-1,1)) 
                     sbox_lo_new(1,1,k,1) = arg_2(i2,j2,nbox-1,l2) + 
     .                      frac* 
     .                  (arg_2(i2,j2,nbox,l2) - arg_2(i2,j2,nbox-1,l2))
                  ENDIF


                  nbox = 1

                  DO WHILE (zaxsrc(1,1,nbox,1) .LT. sbox_hi_lim(1,1,k,1)
     .                      .AND.
     .                  nbox .lt. arg_hi_ss(Z_AXIS,ARG1))
                     nbox = nbox + 1
                  ENDDO

                  IF (nbox .eq. 1) THEN 
                     sbox_hi_new(1,1,k,1) = arg_2(i2,j2,nbox,l2) 
                  ELSE
                     frac = (sbox_hi_lim(1,1,k,1) - 
     .                       zaxsrc(1,1,nbox-1,1))/
     .                  (zaxsrc(1,1,nbox,1) - zaxsrc(1,1,nbox-1,1)) 
                     sbox_hi_new(1,1,k,1) = arg_2(i2,j2,nbox-1,l2) + 
     .                      frac* 
     .                   (arg_2(i2,j2,nbox,l2) - arg_2(i2,j2,nbox-1,l2))
                  ENDIF


  500          CONTINUE 

* For each z line accumulate weighted source points and weights to get
* the average value.   Loop through the each destination grid box.


               sum = 0.
               wtdsum = 0.
               DO 100 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*  Find the first and last source point in this destination box

                  nsrc_lo = 1
                  nsrc_hi = 1

                  DO WHILE 
     .                  (sbox_lo_new(1,1,nsrc_lo,1) .LT. 
     .                      dbox_lo_lim(1,1,k,1) )
                     nsrc_lo = nsrc_lo + 1
                     IF (nsrc_lo .GT. arg_hi_ss(Z_AXIS, ARG2)) GO TO 11
                  ENDDO
  11              CONTINUE

*  Find any portion of the source box lying in the destination box.

                  DO WHILE 
     .                  (sbox_lo_new(1,1,nsrc_hi,1) .LE. 
     .                      dbox_hi_lim(1,1,k,1) )
                     nsrc_hi = nsrc_hi + 1
                     IF (nsrc_hi .GT. arg_hi_ss(Z_AXIS, ARG2)) GO TO 22
                  ENDDO
  22              CONTINUE

                  nsrc_hi = nsrc_hi - 1

*  No source boxes within this destination box.

                  IF (nsrc_hi .LT. nsrc_lo) THEN
                     IF (wtdsum .NE. 0.) THEN
                        result(i,j,k,l) = sum / wtdsum
                        sum = 0.
                        wtdsum = 0.
                     ELSE
                        result(i,j,k,l) = bad_flag_result
                     ENDIF
                     GO TO 160
                  ENDIF

                  DO 90 nsrc = nsrc_lo, nsrc_hi

*  Get fraction of this source box lying within the destination box. 

	               xsrc_min = sbox_lo_new(1,1,nsrc,1)
	               xsrc_max = sbox_hi_new(1,1,nsrc,1)

                       dbox_hi = dbox_hi_lim(1,1,k,1)
                       dbox_lo = dbox_lo_lim(1,1,k,1)

	               xsrc_lo = MAX(xsrc_min, dbox_lo)
	               xsrc_hi = MIN(xsrc_max, dbox_hi)

                       xsrcdiff = xsrc_max - xsrc_min
                       wtd = 0.
                       IF (xsrcdiff .NE. 0.) 
     .                     wtd = (xsrc_hi - xsrc_lo)/xsrcdiff

                       arg1_val = arg_1(i1,j1,nsrc,l1)
                       IF (arg1_val .ne. bad_flag(ARG1)) THEN
	                  sum   = sum  + wtd * arg1_val
                          wtdsum  = wtdsum + wtd

                          arg1_val_last = arg1_val
                          wtd_last = wtd

                       ENDIF
  
  90              CONTINUE

* If we had any good data, calculate the average.

                  IF (wtdsum .NE. 0.) THEN
                     result(i,j,k,l) = sum / wtdsum

*  Start next average with remaining weighted part of the last box with
*  was not within the destination box.

                     sum = (1.- wtd_last)* arg1_val_last
                     wtdsum = 1.- wtd_last
                  ELSE 
                     result(i,j,k,l) = bad_flag_result
                     sum = 0.
                     wtdsum = 0.
                  ENDIF

 160              CONTINUE

 100           CONTINUE

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


