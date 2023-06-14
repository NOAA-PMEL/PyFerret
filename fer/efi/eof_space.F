*
*  eof_space.F
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
*
* Jul 2001 Move to statically linked code in fer/efi. 
* Jul 2001 Move count_eof and pack_eof to file eofsubs.f, rename solve_eof
*          to solve_eof_space.
*          Make the function work w/ data having multiple depths: compute
*          EOF solution for each depth.
* Aug 2001 move to fer/efi directory, to be statically linked to Ferret.  
*	   Change INCLUDE statements to remove directory spec.  
* Feb 2003 Add a test that NEOF <= max_neof, which is dictated by work
*          arrays in the solvers in TRIDIN in eofsubs.F
* Jun 2003 Workspace arrays need to be initialized for each vertical level.
* Sep 2005 Remove the max_eof test; make all workspace be allocated as 
*          external function work arrays.
*
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*  The functions eof_space, eof_stat, eof_tfunc are not implemented in the double
*  precision versions of Ferret (and PyFerret).  Sill, updating the do-loops 
*  in case of future re-implementation of these functions.
*
* Compute EOF/s and time amplitude funcions from a 2-d field, or for each level 
* of a 3-D field.  Based on programs (coadseof.f, etc.) by Dai McClurg and 
* Ansley Manke and calls Billy Kessler's method for finding EOFs  of NON-gappy 
* time series EOFIN, or optionally his implementation of Chelton's '82 method 
* for finding EOFs of gappy time series.
*
* Return Spatial EOF's
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


      SUBROUTINE eof_space_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

* These funtions aren't quite right under double precision. Tolerances
* and epsilons need to be set correctly for double precision computation.

c      CALL ef_set_desc(id,
c     .    'Compute EOF spacial fields from XYT field, gappy in T')
      CALL ef_set_desc(id, 'Function NOT AVAILABLE in this Ferret version')

      CALL ef_set_num_args(id, 2)

      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, ABSTRACT,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 9)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg,
     .                     'Variable in x,y,t; may be fcn of z,e,f')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'frac_timeser')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg,
     . 'Use only those time series with this fraction valid data')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)
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

      SUBROUTINE eof_space_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_lo_l, my_hi_l

      INTEGER nx, ny

*     Use utility functions to get context information about the 
*     1st argument, to set the abstract axis lo and hi indices.

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      nx = arg_hi_ss(X_AXIS, ARG1) - arg_lo_ss(X_AXIS, ARG1) + 1
      ny = arg_hi_ss(Y_AXIS, ARG1) - arg_lo_ss(Y_AXIS, ARG1) + 1

      my_lo_l = 1
      my_hi_l = nx * ny

      CALL ef_set_axis_limits(id, T_AXIS, my_lo_l, my_hi_l)

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
      SUBROUTINE eof_space_work_size(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

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
      COMMON /EOFXYSTOR/ mx
      INTEGER mx1, my1, mt1, mx, mxmt, mxmx

      INTEGER iwork
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mx1 = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      my1 = 1 + arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)
      mt1 = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

      mx = mx1 * my1 + 10
      mxmt = mx * mt1 + 10
      mxmx = mx * mx + 10

*  val
      iwork = 1
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                               mx, 1, 1, 1, 1, 1)

*  taf
      iwork = 2
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                             mxmt, 1, 1, 1, 1, 1)

*  pct
      iwork = 3
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                               mx, 1, 1, 1, 1, 1)

*  vec
      iwork = 4
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                             mxmx, 1, 1, 1, 1, 1)

*  c
      iwork = 5
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                             mxmx, 1, 1, 1, 1, 1)

*  ddat_1d
      iwork = 6
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                             mxmt, 1, 1, 1, 1, 1)

*  isave_jsave
      iwork = 7
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                               mx, 2, 1, 1, 1, 1)

*  ok
      iwork = 8
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1,   1, 1, 1, 1, 1,
     .                              mx1, my1, 1, 1, 1, 1)

*  eofwork
      iwork = 9
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1,  1, 1, 1, 1, 1,
     .                               mx, 10, 1, 1, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this SUBROUTINE we compute the result
*

      SUBROUTINE  eof_space_compute(id, arg_1, arg_2, result, 
     .       val, taf, pct, vec, c, ddat_1d, isave_jsave, ok, eofwork)

*  arg_1  variable, function of (x,y,t)
*  result NX*NY eigenvectors

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

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      COMMON /EOFXYSTOR/ mx
      INTEGER mx

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

*  Dimension the work arrays.

      REAL val(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .         wrk1loz:wrk1hiz, wrk1lot:wrk1hit,
     .         wrk1loe:wrk1hie, wrk1lof:wrk1hif)
      REAL taf(wrk2lox:wrk2hix, wrk2loy:wrk2hiy,
     .         wrk2loz:wrk2hiz, wrk2lot:wrk2hit,
     .         wrk2loe:wrk2hie, wrk2lof:wrk2hif)
      REAL pct(wrk3lox:wrk3hix, wrk3loy:wrk3hiy,
     .         wrk3loz:wrk3hiz, wrk3lot:wrk3hit,
     .         wrk3loe:wrk3hie, wrk3lof:wrk3hif)
      REAL vec(wrk4lox:wrk4hix, wrk4loy:wrk4hiy,
     .         wrk4loz:wrk4hiz, wrk4lot:wrk4hit,
     .         wrk4loe:wrk4hie, wrk4lof:wrk4hif)
      REAL c(wrk5lox:wrk5hix, wrk5loy:wrk5hiy,
     .       wrk5loz:wrk5hiz, wrk5lot:wrk5hit,
     .       wrk5loe:wrk5hie, wrk5lof:wrk5hif)
      REAL ddat_1d(wrk6lox:wrk6hix, wrk6loy:wrk6hiy,
     .             wrk6loz:wrk6hiz, wrk6lot:wrk6hit,
     .             wrk6loe:wrk6hie, wrk6lof:wrk6hif)
      REAL isave_jsave(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .                 wrk7loz:wrk7hiz, wrk7lot:wrk7hit,
     .                 wrk7loe:wrk7hie, wrk7lof:wrk7hif)
      REAL ok(wrk8lox:wrk8hix, wrk8loy:wrk8hiy,
     .        wrk8loz:wrk8hiz, wrk8lot:wrk8hit,
     .        wrk8loe:wrk8hie, wrk8lof:wrk8hif)
      REAL eofwork(wrk9lox:wrk9hix, wrk9loy:wrk9hiy,
     .             wrk9loz:wrk9hiz, wrk9lot:wrk9hit,
     .             wrk9loe:wrk9hie, wrk9lof:wrk9hif)


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

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER neof, ier
      CHARACTER*255 err_msg, err_out

      INTEGER nx, ny, nt, k, m, n, k1, m1, n1,
     .        iw, jw, kw, lw, mw, nw
      REAL frac_timeser
      
      WRITE (err_msg,*)
     .  'Function EOF_SPACE not available in this version of Ferret. Use EOFSVD_SPACE'
      CALL ef_bail_out (id, err_msg)

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*  get array sizes

      nx = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      ny = 1 + arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)
      nt = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

*  Get time percent parameter.

      frac_timeser = arg_2(arg_lo_ss(X_AXIS,ARG2), 
     .                     arg_lo_ss(Y_AXIS,ARG2),
     .                     arg_lo_ss(Z_AXIS,ARG2), 
     .                     arg_lo_ss(T_AXIS,ARG2),
     .                     arg_lo_ss(E_AXIS,ARG2),
     .                     arg_lo_ss(F_AXIS,ARG2))

*  Compute EOF for each depth, ensemble, forecast

      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 300 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)

      m1 = arg_lo_ss(E_AXIS,ARG1)
      DO 200 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

      k1 = arg_lo_ss(Z_AXIS,ARG1)
      DO 100 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

*  Initialize work arrays.

         val = 0.
         taf = 0.
         pct = 0.
         vec = 0.
         c = 0.
         ddat_1d = 0.
         isave_jsave = 0.
         ok = 0.
         eofwork = 0.

*  Find the number of eofs to solve for.
*  neof = number of (x,y) points with frac_timeser good data.  
*  Set OK to mark where they are.   (note this also allows for 
*  working around continental boundaries or other areas where 
*  entire time series are missing)

         CALL count_neof (arg_1, neof, ok, nx, ny, nt, k1, m1, n1,
     .                    arg_lo_ss, arg_hi_ss, arg_incr, mx,
     .                    bad_flag(ARG1), frac_timeser, err_msg, ier)
         IF (ier .NE. 0) then
            GOTO 5010
         ENDIF

*  Put the data into the array ddat_1d(neof,nt) 

         CALL pack_ef (arg_1, ddat_1d, isave_jsave, neof,  
     .                 ok, frac_timeser, nx, ny, nt, k1, m1, n1,
     .                 arg_lo_ss, arg_hi_ss, arg_incr)

*  Solve for the EOF's: eigenvectors, time functions, percent variance explained.

         CALL solve_eof_space (ddat_1d, neof, nt, k, m, n, val, vec, 
     .            taf, pct, c, result, isave_jsave, eofwork, res_lo_ss,
     .            res_hi_ss, res_incr, bad_flag(ARG1), bad_flag_result,
     .            frac_timeser, err_msg, ier)
         IF (ier .NE. 0) GOTO 5020

         k1 = k1 + arg_incr(Z_AXIS,ARG1)
  100 CONTINUE
         m1 = m1 + arg_incr(E_AXIS,ARG1)
  200 CONTINUE
         n1 = n1 + arg_incr(F_AXIS,ARG1)
  300 CONTINUE

      RETURN

 5010 CALL ef_bail_out (id, err_msg)
      RETURN

 5020 WRITE (err_out, 5500) k, m, n, err_msg 
 5500 FORMAT ('k,m,n=', 3I3, A250)
      CALL ef_bail_out (id, err_out)
      RETURN

      END

***********************************************************************

      SUBROUTINE solve_eof_space (ddat_1d, neof, nt, k, m, n, val, vec, 
     .            taf, pct, c, result, isave_jsave, eofwork, res_lo_ss,
     .            res_hi_ss, res_incr, bad_flag_dat, bad_flag_result, 
     .            frac_timeser, err_msg, ier)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

* arguments
      INTEGER neof, nt, k, m, n
      REAL ddat_1d(neof,nt)
      REAL val(*)
      REAL vec(neof, neof)
      REAL taf(neof, nt)
      REAL pct(*)
      REAL c(neof, neof)
      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)
      REAL isave_jsave(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .                 wrk7loz:wrk7hiz, wrk7lot:wrk7hit,
     .                 wrk7loe:wrk7hie, wrk7lof:wrk7hif)
      REAL eofwork(wrk9lox:wrk9hix, wrk9loy:wrk9hiy,
     .             wrk9loz:wrk9hiz, wrk9lot:wrk9hit,
     .             wrk9loe:wrk9hie, wrk9lof:wrk9hif)
      INTEGER res_lo_ss(6), res_hi_ss(6), res_incr(6)
      REAL bad_flag_dat, bad_flag_result, frac_timeser
      CHARACTER*(*) err_msg
      INTEGER ier

* local variables
      INTEGER i, j, l, mm, nn, is, js

      IF (frac_timeser .LT. 1.) THEN
         CALL EOFIN_CHEL_GAP (ddat_1d, neof, nt, val, vec, taf, pct, c, 
     .            eofwork, bad_flag_dat, bad_flag_result, err_msg, ier)
         IF (ier .NE. 0) RETURN
      ELSE
         CALL EOFIN(ddat_1d, neof, nt, val, vec, taf, pct, c, eofwork)
      ENDIF

*  Result is the eigenvectors.

      DO 130 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         DO 120 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
            DO 110 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
               result(i,j,k,l,m,n) = bad_flag_result
  110       CONTINUE
  120    CONTINUE
  130 CONTINUE

      DO 220 nn = 1, neof
         DO 210 mm = 1, neof
            IF (res_lo_ss(X_AXIS) .EQ. ef_unspecified_int4) THEN
               is = ef_unspecified_int4
            ELSE
               is = res_lo_ss(X_AXIS) + isave_jsave(mm,1,1,1,1,1) - 1
            ENDIF

            IF (res_lo_ss(Y_AXIS) .EQ. ef_unspecified_int4) THEN
               js = ef_unspecified_int4
            ELSE
               js = res_lo_ss(Y_AXIS) + isave_jsave(mm,2,1,1,1,1) - 1
            ENDIF

            result(is,js,k,nn,m,n) = vec(mm,nn)

  210    CONTINUE
  220 CONTINUE

      RETURN
      END
