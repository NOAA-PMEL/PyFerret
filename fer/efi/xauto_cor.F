*
* xauto_cor.F
*
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
* July 1999
*  Update to use abstract axis 1-feb-2000
*  acm 11/2001 bug fix: check both arg_1(i1+im...) and arg_1(i1...) for bad flag

* This function computes the autocorrelation for a x series.  
*  Autocorelation formula computed directly for lags of 0, 1, ..., N-1
*  Return on an abstract X axis.

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

      SUBROUTINE xauto_cor_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .        'Compute autocorrelation of series, lags of 0, ..., N-1')
      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT,        IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 3)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_desc(id, arg, 'X-series variable')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, YES, YES, YES, YES, YES)

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
* lo_ss               lo subscript for an axis
*
* hi_ss               hi subscript for an axis
*

      SUBROUTINE xauto_cor_result_limits(id)

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
      INTEGER lo_ss, hi_ss

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

C  Autocorrelation returns correlations for lags of 0 to N-1.
      lo_ss = 1
      hi_ss = arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1) + 1

      CALL ef_set_axis_limits(id, X_AXIS, lo_ss, hi_ss)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************
      RETURN
      END


*
* In this subroutine we request an amount of storage to be supplied
* by Ferret and passed as an additional argument.
*
      SUBROUTINE xauto_cor_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
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
      INTEGER npts, irr

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      npts = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
  
* partial means  pmean
      irr = 1
      CALL ef_set_work_array_dims_6d(id, irr,
     .                                1, 1, 1, 1, 1, 1,
     .                             npts, 1, 1, 1, 1, 1)

* partial variances  pvar
      irr = 2
      CALL ef_set_work_array_dims_6d(id, irr,
     .                                1, 1, 1, 1, 1, 1,
     .                             npts, 1, 1, 1, 1, 1)

* box sizes BOX
      irr = 3
      CALL ef_set_work_array_dims_6d(id, irr,
     .                                1, 1, 1, 1, 1, 1,
     .                             npts, 1, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE xauto_cor_compute(id, arg_1, result,
     .                             pmean, pvar, box)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* Ignore the Y/Z/T/E/F dimensions in the work arrays since
* they are not used and Fortran is column major
      REAL pmean(wrk1lox:wrk1hix)
      REAL pvar (wrk2lox:wrk2hix)
      REAL box  (wrk3lox:wrk3hix)

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

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      REAL psum, diff, diff2
      REAL dsum, bsize
      INTEGER ir, im, is
      INTEGER nd, arg
      INTEGER i, j, k, l, m, n
      INTEGER i1, j1, k1, l1, m1, n1

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      arg = 1
      CALL ef_get_box_size(id, arg, X_AXIS, arg_lo_ss(X_AXIS,arg), 
     .                     arg_hi_ss(X_AXIS,arg), box)

      nd = arg_hi_ss(X_AXIS,arg) - arg_lo_ss(X_AXIS,arg) + 1

      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m1 = arg_lo_ss(E_AXIS,ARG1)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      l1 = arg_lo_ss(T_AXIS,ARG1)
      DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         k1 = arg_lo_ss(Z_AXIS,ARG1)
         DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

            j1 = arg_lo_ss(Y_AXIS,ARG1)
            DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

*
* Calculate the autocorrelation 
*  First step is to compute the partial means [1/(n-i)]* sum[m=1,n-i]dat(m+i)

               DO 120 i = 1, nd
                  psum = 0.
                  dsum = 0.
                  im = i-1
                  i1 = arg_lo_ss(X_AXIS,ARG1)
                  DO 110 is = 1, nd-im 
                     IF ( arg_1(i1+im,j1,k1,l1,m1,n1)
     .                    .NE. bad_flag(ARG1) ) THEN
                        bsize = box(is+im)
                        psum = psum + bsize*arg_1(i1+im,j1,k1,l1,m1,n1)
                        dsum = dsum + bsize
                     ENDIF
                     i1 = i1 + arg_incr(X_AXIS,ARG1)
 110              CONTINUE
                  IF ( dsum .GT. 0. ) pmean(i) = psum / dsum
 120           CONTINUE

*  Compute the partial variances RMS[dat(m+i) - pmean(i)]
*  Don't divide by dsum; the denominators cancel when the autocorrelation
*  is computed below.

               DO 160 i = 1, nd
                  psum = 0.
                  dsum = 0.
                  im = i-1
                  i1 = arg_lo_ss(X_AXIS,ARG1)
                  DO 150 is = 1, nd-im 
                     IF ( arg_1(i1+im,j1,k1,l1,m1,n1)
     .                    .NE. bad_flag(ARG1) ) THEN
                        bsize = box(is+im)
                        diff = arg_1(i1+im,j1,k1,l1,m1,n1) - pmean(i)
                        psum = psum + bsize * diff * diff
                        dsum = dsum + bsize
                     ENDIF
                     i1 = i1 + arg_incr(X_AXIS,ARG1)
 150              CONTINUE
                  IF ( dsum .GT. 0. ) THEN
                     pvar(i) = sqrt(psum)
                  ELSE
                     pvar(i) = 0.
                  ENDIF
 160           CONTINUE

*  Compute the autocorrelation for lag l=0,1,2,...,ND-1

               i1 = arg_lo_ss(X_AXIS,ARG1)
               ir = res_lo_ss(X_AXIS)
               DO 180 i = 1, nd

                  psum = 0.
                  dsum = 0.
                  im = i-1
                  i1 = arg_lo_ss(X_AXIS,ARG1)
                  DO 170 is = 1, nd-im 
                     IF ( arg_1(i1+im,j1,k1,l1,m1,n1)
     .                    .NE. bad_flag(ARG1) .AND.
     .                    arg_1(i1,   j1,k1,l1,m1,n1)    
     .                    .NE. bad_flag(ARG1) ) THEN
                        bsize = (box(is+im) + box(is)) / 2.
                        diff  = arg_1(i1+im,j1,k1,l1,m1,n1) - pmean(i)
                        diff2 = arg_1(i1,   j1,k1,l1,m1,n1) - pmean(1)
                        psum = psum + bsize * diff * diff2
                        dsum = dsum + box(is+im) * box(m)
                     ENDIF
                     i1 = i1 + arg_incr(X_AXIS,ARG1)
 170              CONTINUE

                  IF ( pvar(1) .NE. 0.   .AND.  
     .                 pvar(i) .NE. 0.   .AND. 
     .                 psum    .NE. 0. ) THEN
                     result(ir,j,k,l,m,n) = psum / (pvar(1) * pvar(i))
                  ELSE
                     result(ir,j,k,l,m,n) = bad_flag_result
                  ENDIF

                  i1 = i1 + arg_incr(X_AXIS,ARG1)
                  ir = ir + res_incr(X_AXIS)
 180           CONTINUE

               j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200        CONTINUE

            k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300     CONTINUE

         l1 = l1 + arg_incr(T_AXIS,ARG1)
 400  CONTINUE

         m1 = m1 + arg_incr(E_AXIS,ARG1)
 500  CONTINUE

         n1 = n1 + arg_incr(F_AXIS,ARG1)
 600  CONTINUE

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
