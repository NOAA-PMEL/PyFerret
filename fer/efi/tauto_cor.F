*
* tauto_cor.F
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
*
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

      SUBROUTINE tauto_cor_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg
      CHARACTER*110 fcn_desc


c      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      WRITE (fcn_desc, 20)
      CALL ef_set_desc(id, fcn_desc)
   20 FORMAT ( 'Compute autocorrelation of ',
     .        ' series, lags of 0, ..., N-1' )

      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 1)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .        IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, ABSTRACT)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 3)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_desc(id, arg, 'X- series variable')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, NO)
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

      SUBROUTINE tauto_cor_result_limits(id)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER nt, nt1

      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

*     Use utility functions to get context information about the argument.

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      nt = arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1) +1

C  Autocorrelation returns corelations for lags of 1 to N.

      nt1 = 1

      CALL ef_set_axis_limits(id, T_AXIS, nt1, nt)

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
      SUBROUTINE tauto_cor_work_size(id)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER npts, irr

      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      npts = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)
  
* partial means  pmean
      irr = 1
      CALL ef_set_work_array_dims (id, irr, 1, 1, 1, 1, npts, 1, 1, 1)


* partial variances  pvar
      irr = 2
      CALL ef_set_work_array_dims (id, irr, 1, 1, 1, 1, npts, 1, 1, 1)

* box sizes BOX
      irr = 3
      CALL ef_set_work_array_dims (id, irr, 1, 1, 1, 1, npts, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE tauto_cor_compute(id, arg_1, result, pmean, 
     .           pvar, box)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      REAL psum, diff, diff2
      REAL dsum, bsize
      INTEGER lr, lm, m

*  Dimension work arrays

      REAL pmean(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL pvar(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL box(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)


      INTEGER nd, arg
      INTEGER i, j, k, l
      INTEGER i1, j1, k1, l1

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      arg = 1
      CALL ef_get_box_size(id, arg, T_AXIS, arg_lo_ss(T_AXIS,arg), 
     .                     arg_hi_ss(T_AXIS,arg), box)

      nd = (arg_hi_ss(T_AXIS,arg) - arg_lo_ss(T_AXIS,arg) + 1)

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*
* Calculate the autocorrelation 
*  First step is to compute the partial means [1/(n-i)]* sum[m=1,n-i]dat(m+i)

               DO 120 l = 1, nd
                  psum = 0.
                  dsum = 0.
                  lm = l-1
                  l1 = arg_lo_ss(T_AXIS,ARG1)
                  DO 110 m = 1, nd-lm 
                     IF (arg_1(i1,j1,k1,l1+lm) .NE. bad_flag(ARG1)) THEN
                        bsize = box(m+lm,1,1,1)
                        psum = psum + bsize* arg_1(i1,j1,k1,l1+lm)
                        dsum = dsum + bsize
                     ENDIF

                     l1 = l1 + arg_incr(T_AXIS,ARG1)
 110              CONTINUE
                  IF (dsum .gt. 0.) pmean(l,1,1,1) = psum/ dsum

 120           CONTINUE

*  Compute the partial variances RMS[dat(m+l) - pmean(l)]
*  Don't divide by dsum; the denominators cancel when the autocorrelation
*  is computed below.

               DO 160 l = 1, nd
                  psum = 0.
                  dsum = 0.

                  lm = l-1
                  l1 = arg_lo_ss(T_AXIS,ARG1)
                  DO 150 m = 1, nd-lm 
                     IF (arg_1(i1,j1,k1,l1+lm) .NE. bad_flag(ARG1)) THEN
                        bsize = box(m+lm,1,1,1)
                        diff = 
     .                     arg_1(i1,j1,k1,l1+lm) - pmean(l,1,1,1)
                        psum = psum + bsize* diff* diff
                        dsum = dsum + bsize

                     ENDIF

                     l1 = l1 + arg_incr(T_AXIS,ARG1)
 150              CONTINUE

                  IF (dsum .gt. 0.) THEN
                     pvar(l,1,1,1) = sqrt(psum)
                  ELSE
                     pvar(l,1,1,1) = 0.
                  ENDIF

 160           CONTINUE

*  Compute the autocorrelation for lag l=0,1,2,...,ND-1

               l1 = arg_lo_ss(T_AXIS,ARG1)
               lr = res_lo_ss(T_AXIS)
               DO 180 l = 1, nd

                  psum = 0.
                  dsum = 0.
                  lm = l-1
                  l1 = arg_lo_ss(T_AXIS,ARG1)
                  DO 170 m = 1, nd-lm 
                     IF (arg_1(i1,j1,k1,l1+lm) .NE. bad_flag(ARG1)) THEN
                        bsize = (box(m+lm,1,1,1) + box(m,1,1,1))/ 2.
                        diff = 
     .                     arg_1(i1,j1,k1,l1+lm) - pmean(l,1,1,1)
                        diff2 = 
     .                     arg_1(i1,j1,k1,l1)    - pmean(1,1,1,1)
                        psum = psum + bsize* diff* diff2
                        dsum = dsum + box(m+lm,1,1,1)* box(m,1,1,1)
                     ENDIF

                     l1 = l1 + arg_incr(T_AXIS,ARG1)
 170              CONTINUE

                  IF (pvar(1,1,1,1) .NE. 0.   .AND.  
     .                pvar(l,1,1,1) .NE. 0.   .AND. 
     .                psum .NE. 0.) THEN
     .                     
                     result(i,j,k,lr) = psum/ 
     .                                (pvar(1,1,1,1)* pvar(l,1,1,1))

                  ELSE
                     result(i,j,k,lr) = bad_flag_result
                  ENDIF

                  l1 = l1 + arg_incr(T_AXIS,ARG1)
                  lr = lr + res_incr(T_AXIS)
 180           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 200        CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 400  CONTINUE


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END
