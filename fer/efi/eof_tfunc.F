*
*  eof_tfunc.F
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
* Sep 1998
* Aug 2000 change fcn description; argument can be fcn of Z as well as x,y,t
* May 2001 Use simpler solver EOFIN when frac_timeser = 1.
* Jul 2001 Move to statically linked code in fer/efi.
* Jul 2001 move count_eof and pack_eof to file eofsubs.f, rename solve_eof
*          to solve_eof_tfunc.
*          Make the function work w/ data having multiple depths: compute
*          EOF solution for each depth.
* Aug 2001 move to fer/efi directory, to be statically linked to Ferret.  
*	   Change INCLUDE statements to remove directory spec.
* Feb 2003 Add a test that NEOF <= max_neof, which is dictated by work
*          arrays in the solvers in TRIDIN in eofsubs.F
* Jun 2003 Workspace arrays need to be initialized for each vertical level.
*
* Compute EOF/s and time amplitude funcions from a 2-d field, or for each level 
* of a 3-D field.  Based on programs (coadseof.f, etc.) by Dai McClurg and 
* Ansley Manke and calls Billy Kessler's method for finding EOFs  of NON-gappy 
* time series EOFIN, or optionally his implementation of Chelton's '82 method 
* for finding EOFs of gappy time series.
*
* Return time amplitude functions.
*
*
* In this SUBROUTINE we provide information about
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


      SUBROUTINE eof_tfunc_init(id)

      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CALL ef_set_desc(id,
     .   'Compute EOF time functions from XYT field.')

      CALL ef_set_num_args(id, 2)

      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .      NORMAL, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 9)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg,'Variable in x,y,t; may be fcn of z')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'frac_timeser')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg,
     . 'Use only those time series with this fraction valid data')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

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

      SUBROUTINE eof_tfunc_result_limits(id)

      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo_l, my_hi_l

      INTEGER nx, ny

*     Use utility functions to get context information about the 
*     1st argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG1) - arg_lo_ss(X_AXIS, ARG1) + 1
      ny = arg_hi_ss(Y_AXIS, ARG1) - arg_lo_ss(Y_AXIS, ARG1) + 1

      my_lo_l = 1
      my_hi_l = nx * ny

      CALL ef_set_axis_limits(id, X_AXIS, my_lo_l, my_hi_l)

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
      SUBROUTINE eof_tfunc_work_size(id)

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
      COMMON /eof_tfuncSTOR/ mx
      INTEGER mx1, my1, mt1, mx, mxmt, mxmx, mxmy

      INTEGER iwork
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mx1 = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      my1 = 1 + arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)
      mt1 = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

      mxmy = mx1 * my1 + 10
      mxmt = mxmy * mt1 + 10
      mxmx = mxmy* mxmy + 10

*  val
      iwork = 1
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mxmy, 1, 1, 1)

*  taf
      iwork = 2
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mxmt, 1, 1, 1)

*  pct
      iwork = 3
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mxmy, 1, 1, 1)

*  vec
      iwork = 4
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mxmx, 1, 1, 1)

*  c
      iwork = 5
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mxmx, 1, 1, 1)

*  ddat_1d
      iwork = 6
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mxmt, 1, 1, 1)

*  isave_jsave
      iwork = 7
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mxmy, 2, 1, 1)

*  ok
      iwork = 8
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mx1, my1, 1, 1)

*  eofwork
      iwork = 9
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mxmy, 10, 1, 1)

      mx = mxmy

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this SUBROUTINE we compute the result
*

      SUBROUTINE  eof_tfunc_compute(id, arg_1, arg_2, result, 
     .       val, taf, pct, vec, c, ddat_1d, isave_jsave, ok, eofwork)

*  arg_1  variable, function of (x,y,t)
*  result neof*NT time amplitude functions

*  Work arrays:
*  val, taf, pct, vec, c, ddat_1d, isave_jsave, ok

*  val(neof)	eigenvalues (Lambda)
*  result == vec(neof,neof) eigenvectors (Lambda*U). Same units as data.
*  taf(neof,nt)	time amplitude functions (V). Dimensionless.
*  pct(neof)	% variance represented by each EOF.
*  c(neof,neof)	work space for cov matrix (garbage output)
*  isave_jsave  save the locations of the data in the x-y plane
*  eofwork	used by QRSYM and other solver routines; replaces
*		original ALPHA, BETA, BB, and P arrays.


      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      COMMON /eof_tfuncSTOR/ mx
      INTEGER mx

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

*  Dimension the work arrays.

      REAL val(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)
      REAL taf(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .               wrk2loz:wrk2hiz, wrk2lot:wrk2hit)
      REAL pct(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .               wrk3loz:wrk3hiz, wrk3lot:wrk3hit)
      REAL vec(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .               wrk4loz:wrk4hiz, wrk4lot:wrk4hit)
      REAL c(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .               wrk5loz:wrk5hiz, wrk5lot:wrk5hit)

      REAL ddat_1d(wrk6lox:wrk6hix, wrk6loy:wrk6hiy,
     .               wrk6loz:wrk6hiz, wrk6lot:wrk6hit)
      REAL isave_jsave(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .               wrk7loz:wrk7hiz, wrk7lot:wrk7hit)
      REAL ok(wrk8lox:wrk8hix, wrk8loy:wrk8hiy,
     .               wrk8loz:wrk8hiz, wrk8lot:wrk8hit)
      REAL eofwork(wrk9lox:wrk9hix, wrk9loy:wrk9hiy,
     .               wrk9loz:wrk9hiz, wrk9lot:wrk9hit)


* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS), arg_hi_ss(4,EF_MAX_ARGS),
     .     arg_incr(4,EF_MAX_ARGS)


************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER neof, max_neof, ier
      CHARACTER*255 err_msg, err_out

      INTEGER nx, ny, nt, i, j, k, l, k1, iw, jw, kw, lw
      REAL frac_timeser

c      DATA max_neof/4000/        ! size of work arrays within the EOF solver


      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  get array sizes

      nx = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      ny = 1 + arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)
      nt = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

*  Get time percent parameter.

      frac_timeser = arg_2(arg_lo_ss(X_AXIS,ARG2), 
     .            arg_lo_ss(Y_AXIS,ARG2), arg_lo_ss(Z_AXIS,ARG2), 
     .            arg_lo_ss(T_AXIS,ARG2))


*  Compute EOF for each depth

      k1 = arg_lo_ss(Z_AXIS,ARG1)
      DO 100 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*  Initialize work arrays.

         DO 701 iw = wrk1lox,wrk1hix
         DO 701 jw = wrk1loy,wrk1hiy
         DO 701 kw = wrk1loz,wrk1hiz 
         DO 701 lw = wrk1lot,wrk1hit
            val(iw,jw,kw,lw) = 0.
  701    CONTINUE

         DO 702 iw = wrk2lox,wrk2hix
         DO 702 jw = wrk2loy,wrk2hiy
         DO 702 kw = wrk2loz,wrk2hiz 
         DO 702 lw = wrk2lot,wrk2hit
            taf(iw,jw,kw,lw) = 0.
  702    CONTINUE

         DO 703 iw = wrk3lox,wrk3hix
         DO 703 jw = wrk3loy,wrk3hiy
         DO 703 kw = wrk3loz,wrk3hiz 
         DO 703 lw = wrk3lot,wrk3hit
            pct(iw,jw,kw,lw) = 0.
  703    CONTINUE

         DO 704 iw = wrk4lox,wrk4hix
         DO 704 jw = wrk4loy,wrk4hiy
         DO 704 kw = wrk4loz,wrk4hiz 
         DO 704 lw = wrk4lot,wrk4hit
            vec(iw,jw,kw,lw) = 0.
  704    CONTINUE

         DO 705 iw = wrk5lox,wrk5hix
         DO 705 jw = wrk5loy,wrk5hiy
         DO 705 kw = wrk5loz,wrk5hiz 
         DO 705 lw = wrk5lot,wrk5hit
            c(iw,jw,kw,lw) = 0.
  705    CONTINUE

         DO 706 iw = wrk6lox,wrk6hix
         DO 706 jw = wrk6loy,wrk6hiy
         DO 706 kw = wrk6loz,wrk6hiz 
         DO 706 lw = wrk6lot,wrk6hit
            ddat_1d(iw,jw,kw,lw) = 0.
  706    CONTINUE

         DO 707 iw = wrk7lox,wrk7hix
         DO 707 jw = wrk7loy,wrk7hiy
         DO 707 kw = wrk7loz,wrk7hiz 
         DO 707 lw = wrk7lot,wrk7hit
            isave_jsave(iw,jw,kw,lw) = 0.
  707    CONTINUE

         DO 708 iw = wrk8lox,wrk8hix
         DO 708 jw = wrk8loy,wrk8hiy
         DO 708 kw = wrk8loz,wrk8hiz 
         DO 708 lw = wrk8lot,wrk8hit
            ok(iw,jw,kw,lw) = 0.
  708    CONTINUE

         DO 709 iw = wrk9lox,wrk9hix
         DO 709 jw = wrk9loy,wrk9hiy
         DO 709 kw = wrk9loz,wrk9hiz 
         DO 709 lw = wrk9lot,wrk9hit
            eofwork(iw,jw,kw,lw) = 0.
  709    CONTINUE

*  Find the number of EOFS to solve for.
*  neof = number of (x,y) points with frac_timeser good data.  
*  Set OK to mark where they are.  (note this also allows for 
*  working around continental boundaries or other areas where 
*  entire time series are missing)

         CALL count_neof (id, arg_1, neof, ok, nx, ny, k1, nt, mx, 
     .       bad_flag(ARG1), frac_timeser, err_msg, ier)

c         IF (neof .GT. max_neof) THEN
c            WRITE (err_msg, 5600) neof, max_neof 
c 5600       FORMAT ('Number of EOFs too large, ', I8, 
c     .              'Reduce NX*NY. Max is', i5)
c            ier = 1
c         ENDIF

         IF (ier .ne. 0) then
            GOTO 5010
         ENDIF

*  Put the data into the array ddat_1d(neof,nt) 

         CALL pack_ef (id, arg_1, ddat_1d, isave_jsave, neof,  
     .                 ok, frac_timeser, nx, ny, k1, nt)

*  Solve for the EOF's: eigenvectors, time functions, percent variance explained.

         CALL solve_eof_tfunc (ddat_1d, neof, nt, k, val, vec, 
     .      taf, pct, c, result, isave_jsave, eofwork, bad_flag(ARG1),
     .      bad_flag_result, frac_timeser, err_msg, ier)
         IF (ier .ne. 0) GOTO 5020

         k1 = k1 + arg_incr(Z_AXIS,ARG1)
  100 CONTINUE

      RETURN

 999  CALL ef_bail_out (id, err_msg)
      RETURN

 5010 CALL ef_bail_out (id, err_msg)
      RETURN

 5020 WRITE (err_out, 5500) k1, err_msg 
 5500 FORMAT ('k=', I3, A250)
      CALL ef_bail_out (id, err_out)
      RETURN

      END

      SUBROUTINE solve_eof_tfunc (ddat_1d, neof, nt, k, val, vec, 
     .      taf, pct, c, result, isave_jsave, eofwork, bad_flag_dat, 
     .      bad_flag_result, frac_timeser, err_msg, ier)


      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'
      INTEGER id

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

      INTEGER neof, nt, ier
      INTEGER i, j, k, l, n

      REAL bad_flag_dat, bad_flag_result, frac_timeser

      REAL isave_jsave(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .               wrk7loz:wrk7hiz, wrk7lot:wrk7hit)
      REAL eofwork(wrk9lox:wrk9hix, wrk9loy:wrk9hiy,
     .               wrk9loz:wrk9hiz, wrk9lot:wrk9hit)

      REAL val(*)
      REAL taf(neof, nt)
      REAL pct(*)
      REAL vec(neof, neof)
      REAL c(neof, neof)

      REAL ddat_1d(neof,nt)

      CHARACTER*(*) err_msg

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)

      IF (frac_timeser .LT. 1.) THEN
         CALL EOFIN_CHEL_GAP (ddat_1d, neof, nt, val, vec, taf, pct, c, 
     .      eofwork, bad_flag_dat, bad_flag_result, err_msg, ier)
      ELSE
	CALL EOFIN(ddat_1d,neof,NT,VAL,VEC,TAF,PCT,C,eofwork)
      ENDIF

      IF (ier .ne. 0) GOTO 999

*  Result is the time functions. 

      j = res_lo_ss(Y_AXIS)
      DO 200 i = 1, neof
         n = 1
         DO 100 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            result(i,j,k,l) = taf(i,n)
            n = n + 1
  100    CONTINUE
  200 CONTINUE

      DO 220 i = neof+1, res_hi_ss(X_AXIS)
         DO 110 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
            result(i,j,k,l) = bad_flag_result
  110    CONTINUE
  220 CONTINUE


      RETURN

  999 CONTINUE
      RETURN

      END
