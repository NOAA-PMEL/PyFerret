*
*  eof_tfunc.F
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
*
* Compute EOF/s and time amplitude funcions from a 2-d field.  Based on 
* program (coadseof.f, etc.) by Dai McClurg and Ansley Manke and calls 
* Billy Kessler's implementation of Chelton's '82 method for finding EOFs 
* of gappy time series.
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

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CHARACTER*100 argdescr

      CALL ef_set_desc(id,
     .  'Compute EOF time functions from x-y-time field w/gaps' )

      CALL ef_set_num_args(id, 3)

      CALL ef_set_axis_inheritance(id, ABSTRACT, 
     .      NORMAL, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 9)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')
      CALL ef_set_arg_desc(id, arg, 'Variable in x,y,t')
      CALL ef_set_axis_influence(id, arg, NO, NO, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'pct_cutoff')
      CALL ef_set_arg_unit(id, arg, ' ')

      WRITE (argdescr, 10)
   10 FORMAT ('(beta)Return EOFs up to cutoff for % variance explained',
     .        ' e.g. 2 for 2%')
      CALL ef_set_arg_desc(id, arg,argdescr)
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
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

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

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

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

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
      COMMON /STOR/ mx
      INTEGER mx1, my1, mt1, mx, mxmt, mxmx

      INTEGER iwork
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      mx1 = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      my1 = 1 + arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)
      mt1 = 1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

      mx = mx1 * my1 + 10
      mxmt = mx * mt1 + 10
      mxmx = mx* mx + 10

*  val
      iwork = 1
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mx, 1, 1, 1)

*  taf
      iwork = 2
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mxmt, 1, 1, 1)

*  pct
      iwork = 3
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mx, 1, 1, 1)

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
     .   mx, 2, 1, 1)

*  ok
      iwork = 8
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mx1, my1, 1, 1)

*  eofwork
      iwork = 9
      CALL ef_set_work_array_dims (id, iwork, 1, 1, 1, 1, 
     .   mx, 4, 1, 1)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END




*
* In this SUBROUTINE we compute the result
*

      SUBROUTINE  eof_tfunc_compute(id, arg_1, arg_2, arg_3, result, 
     .       val, taf, pct, vec, c, ddat_1d, isave_jsave, ok, eofwork)

*  arg_1  variable, function of (x,y,t)
*  arg_2  Cutoff of percent variance explained by eigenvectors:  up 
*         to this cutoff, normalize the eigenvectors and compute time 
*         amplitude functions.
*  arg_3  Cutoff  percent good time series data for point to be included.
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


      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      COMMON /STOR/ mx
      INTEGER mx

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .     mem3lot:mem3hit)

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

      INTEGER neof, ier
      CHARACTER*255 err_msg

      INTEGER nx, ny, nt
      REAL pct_cutoff, frac_timeser

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

C  get array sizes

      nx  =  1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      ny  =  1 + arg_hi_ss(Y_AXIS,ARG1) - arg_lo_ss(Y_AXIS,ARG1)
      nt  =  1 + arg_hi_ss(T_AXIS,ARG1) - arg_lo_ss(T_AXIS,ARG1)

*  Get cutoff parameter.

      pct_cutoff = arg_2(arg_lo_ss(X_AXIS,ARG2), 
     .            arg_lo_ss(Y_AXIS,ARG2), arg_lo_ss(Z_AXIS,ARG2), 
     .            arg_lo_ss(T_AXIS,ARG2))

*  Get time percent parameter.

      frac_timeser = arg_3(arg_lo_ss(X_AXIS,ARG3), 
     .            arg_lo_ss(Y_AXIS,ARG3), arg_lo_ss(Z_AXIS,ARG3), 
     .            arg_lo_ss(T_AXIS,ARG3))


*  Find the number of eofs to solve for.
*  neof = number of (x,y) points with some  good data.  
*  Set OK to mark where they are.

      CALL count_neof (arg_1, neof, ok, nx, ny, nt, mx, 
     .       bad_flag_result, frac_timeser, err_msg, ier)

      IF (ier .ne. 0) then
         GOTO 999
      ENDIF

*  Put the data into the array ddat_1d(neof,nt) 

      CALL pack_ef (arg_1, ddat_1d, isave_jsave, neof,  
     .                 ok, frac_timeser, nx, ny, nt)

*  Solve for the EOF's: eigenvectors, time functions, percent variance explained.

      CALL solve_eof (ddat_1d, neof, nt, val, vec, taf, pct, c, 
     .      result, isave_jsave, eofwork, bad_flag_result, 
     .      pct_cutoff, err_msg, ier)
      IF (ier .ne. 0) GOTO 999

      RETURN

 999  CALL ef_bail_out (id, err_msg)
      RETURN

      END


      SUBROUTINE solve_eof (ddat_1d, neof, nt, val, vec, taf, pct, c, 
     .      result, isave_jsave, eofwork, bad_flag_result, pct_cutoff, 
     .      err_msg, ier)


      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'
      INTEGER id

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

      INTEGER neof, nt, nout, ier
      INTEGER i, j, k, n

      REAL bad_flag_result, pct_cutoff

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

      CALL EOFIN_CHEL_GAP (ddat_1d, neof, nt, val, vec, taf, pct, c, 
     .      eofwork, pct_cutoff, nout, bad_flag_result, err_msg, ier)

      IF (ier .ne. 0) GOTO 999


*  Result is the time functions, as many as we save when pct_cutoff applied.

      DO 400 J = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)
         DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
            DO 200 i = 1, nout
               DO 100 n = 1, nt
                  result(i,j,k,n) = taf(i,n)
  100          CONTINUE
  200       CONTINUE

            DO 220 i = nout+1, res_hi_ss(X_AXIS)
               DO 110 n = 1, nt
                  result(i,j,k,n) = bad_flag_result
  110          CONTINUE
  220       CONTINUE

  300    CONTINUE
  400 CONTINUE

      RETURN


  999 CONTINUE

      
      RETURN

      END

      
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************
************************************************************************
      SUBROUTINE count_neof (ddat, neof, ok, nx, ny, nt, mx, 
     .      flag, frac_timeser, err_msg, ier)

*  Find number of time series with valid data.  This will be neof.
*  Set array OK to mark which (x,y) have some good data.

      INTEGER nt, i, j, n, nx, ny
      INTEGER mx, neof, ier

      REAL ddat(nx,ny,1,nt)
      REAL flag, a_nt, frac_timeser

      REAL ok(nx,ny)
      CHARACTER*(*) err_msg

*  Set array ok = 1 if frac_timeser fraction of valid data exists at the point.
 
      a_nt = float(nt)

      neof = 0
      DO j = 1, ny
         DO i = 1, nx
            ok(i,j) = 0.
            DO n = 1, nt
               IF (ddat(i,j,1,n) .ne. flag) then
                  ok(i,j) = ok(i,j) + 1.
               endif
            ENDDO
            ok(i,j) = ok(i,j)/ a_nt
            IF (ok(i,j) .ge. frac_timeser) neof = neof + 1  
         ENDDO
      ENDDO

      ier = 0
      IF (neof .gt. mx) then
        WRITE(err_msg,*) 'Increase parameter mx in eof.F ',
     .                   'Set mx at least', neof
        ier = 1
        RETURN
      ENDIF

      RETURN
      END

************************************************************************
      SUBROUTINE pack_ef (ddat, ddat_1d, isave_jsave, neof,  
     .                 ok, frac_timeser, nx, ny, nt)

*  put ddat(x,y,t) into one list ddat_1d(neof,nt)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER nt, i, j, n, nx, ny
      INTEGER ipack, neof

      REAL ddat(nx,ny,1,nt), ddat_1d(neof,nt)

      REAL isave_jsave(wrk7lox:wrk7hix, wrk7loy:wrk7hiy,
     .               wrk7loz:wrk7hiz, wrk7lot:wrk7hit)

      REAL frac_timeser
      REAL ok(nx,ny)

      ipack = 0

      DO j = 1, ny
        DO i = 1, nx
          IF (ok(i,j) .ge. frac_timeser) then
            ipack = ipack + 1

            isave_jsave(ipack,1,1,1) = i
            isave_jsave(ipack,2,1,1) = j
 
            DO n = 1, nt
              ddat_1d(ipack, n) = ddat(i,j,1,n) 
            ENDDO

          endif
        ENDDO
      ENDDO

      return
      end
