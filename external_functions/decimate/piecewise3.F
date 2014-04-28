*  piecewise3.F
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

* Ansley Manke June 2013
* implements stl2, Piecewise linear approximations of fewest line segments
* described in http://www.computer.org/csdl/proceedings/afips/1972/5079/00/50790187.pdf
* and a somewhat later version of the code, found at
* http://www.netlib.org/toms/510
*
*
*  In this subroutine we provide information about
*  the function.  The user configurable information 
*  consists of the following:
* 
*  descr              Text description of the function
* 
*  num_args           Required number of arguments
* 
*  axis_inheritance   Type of axis for the result
*                        ( CUSTOM, IMPLIED_BY_ARGS, NORMAL, ABSTRACT )
*                        CUSTOM          - user defined axis
*                        IMPLIED_BY_ARGS - same axis as the incoming argument
*                        NORMAL          - the result is normal to this axis
*                        ABSTRACT        - an axis which only has index values
* 
*  piecemeal_ok       For memory optimization:
*                        axes where calculation may be performed piecemeal
*                        ( YES, NO )
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


      SUBROUTINE piecewise3_init(id)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Return indices to sample from piecewise linear interpolation of 3 vars' )

      CALL ef_set_num_args(id, 7) 
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, ABSTRACT, NORMAL, 
     .                                              NORMAL,   NORMAL)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 6)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'V1')
      CALL ef_set_arg_desc(id, arg, '1st variable to sample')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'V2')
      CALL ef_set_arg_desc(id, arg, '2nd variable to sample')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'V3')
      CALL ef_set_arg_desc(id, arg, '3rd variable to sample')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'Control parameter')
      CALL ef_set_arg_desc(id, arg,
     .          'Control parameter (e.g. 6 for ntol=rest=cont=1)')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'Tol1')
      CALL ef_set_arg_desc(id, arg, 'Tolerance for V1')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'Tol2')
      CALL ef_set_arg_desc(id, arg, 'Tolerance for V2')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'Tol3')
      CALL ef_set_arg_desc(id, arg, 'Tolerance for V3')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)



*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

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

      SUBROUTINE piecewise3_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      INTEGER my_hi_i
      INTEGER arg_lo_ss(4,EF_MAX_ARGS),
     .        arg_hi_ss(4,EF_MAX_ARGS),
     .        arg_incr (4,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the arguments.
*

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      my_hi_i = arg_hi_ss(X_AXIS,1) - arg_lo_ss(X_AXIS,1) + 1

      CALL ef_set_axis_limits(id, X_AXIS, 1, my_hi_i)
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
      SUBROUTINE piecewise3_work_size(id)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays, X/Y/Z/T/E/F dimensions
*
* ef_set_work_array_dims(id, array #,
*                           xlo, ylo, zlo, tlo, 
*                           xhi, yhi, zhi, thi)
*

      INTEGER arg_lo_ss(4,EF_MAX_ARGS),
     .        arg_hi_ss(4,EF_MAX_ARGS),
     .        arg_incr (4,EF_MAX_ARGS)
      INTEGER npts, iwork

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      npts = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
      npts = 2*npts

* input x-axis coordinates
      iwork = 1
      CALL ef_set_work_array_dims(id, iwork,
     .                                1, 1, 1, 1,
     .                             npts+10, 1, 1, 1)

* internal computation, u1 for y1
      iwork = 2
      CALL ef_set_work_array_dims(id, iwork,
     .                                1, 1, 1, 1,
     .                             npts+10, 1, 1, 1)

* internal computation, u2 for y2
      iwork = 3
      CALL ef_set_work_array_dims(id, iwork,
     .                                1, 1, 1, 1,
     .                             npts+10, 1, 1, 1)

* internal computation, u3 for y3
      iwork = 4
      CALL ef_set_work_array_dims(id, iwork,
     .                                1, 1, 1, 1,
     .                             npts+10, 1, 1, 1)

* internal computation, v
      iwork = 5
      CALL ef_set_work_array_dims(id, iwork,
     .                                1, 1, 1, 1,
     .                           2*npts+10, 1, 1, 1)

* internal computation, w
      iwork = 6
      CALL ef_set_work_array_dims(id, iwork,
     .                                1, 1, 1, 1,
     .                           2*npts+10, 1, 1, 1)
*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*  In this subroutine we compute the result
* 
      SUBROUTINE piecewise3_compute(id, arg_1, arg_2, arg_3, arg_4, 
     .        arg_5, arg_6, arg_7, result, xpts, u1, u2, u3, v, w)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .           mem3lot:mem3hit)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .           mem4lot:mem4hit)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz, 
     .           mem5lot:mem5hit)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, mem6loz:mem6hiz, 
     .           mem6lot:mem6hit)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, mem7loz:mem7hiz, 
     .           mem7lot:mem7hit)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit)

* Ignore the Y/Z/T/E/F dimensions in the work array since
* they are not used and Fortran is column major

      REAL xpts(wrk1lox:wrk1hix)

      REAL u1(wrk2lox:wrk2hix)
      REAL u2(wrk3lox:wrk3hix)
      REAL u3(wrk4lox:wrk4hix)
      REAL  v(wrk5lox:wrk5hix)
      REAL  w(wrk6lox:wrk6hix)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4),
     .        res_hi_ss(4),
     .        res_incr (4)
      INTEGER arg_lo_ss(4,EF_MAX_ARGS),
     .        arg_hi_ss(4,EF_MAX_ARGS),
     .        arg_incr (4,EF_MAX_ARGS)

      REAL bad_flag(EF_MAX_ARGS), bad_flag_result


* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      INTEGER i,  k
      INTEGER nx, nout1, nout2, nout3, nmerge1, nmerge2, ip
      REAL val, tol(3), bad_y
      CHARACTER*100 errtxt

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* Initialize
      nx = 0
      nout1 = 0
      nout2 = 0
      nout3 = 0
      nmerge1 = 0
      nmerge2 = 0

* Control parameter IP

      CALL ef_get_one_val(id, ARG4, val)
      IP = INT(val)

* Tolerances

      CALL ef_get_one_val(id, ARG5, tol(1))

      CALL ef_get_one_val(id, ARG6, tol(2))

      CALL ef_get_one_val(id, ARG7, tol(3))

*  Set x coordinate - abstract axis.

      nx = arg_hi_ss(X_AXIS, ARG1) - arg_lo_ss(X_AXIS, ARG1) + 1
      DO 50 i = 1, nx
         xpts(i) = i
  50  CONTINUE

      k = res_lo_ss(Z_AXIS)  ! index for unused dimensions

* Find piecewise fit for each array then merge the indices
* Set the work arrays to 0's so we arent using spurious data
* If the function returns too many points, just return all indices


* STL2 modified to truncate result to next lowest integer

      bad_y = bad_flag(ARG1)
      CALL STL2(xpts, arg_1, tol(1), nx, bad_y, bad_flag_result, 
     .   u1, v, w, nout1, IP)
      IF (nout1 .GT. res_hi_ss(X_AXIS)) THEN
         nout1 = nx
         DO 100 i = 1, nx
	    u1(i) = FLOAT(i)
 100     CONTINUE
      ENDIF
      
      DO 110 i = nout1+1, wrk2hix
         u1(i) = 0.
 110  CONTINUE

      bad_y = bad_flag(ARG2)
      CALL STL2(xpts, arg_2, tol(2), nx, bad_y, bad_flag_result, 
     .   u2, v, w, nout2, IP)
      IF (nout2 .GT. res_hi_ss(X_AXIS)) THEN
         nout2 = nx
         DO 200 i = 1, nx
	    u2(i) = FLOAT(i)
 200     CONTINUE
      ENDIF

      DO 210 i = nout2+1, wrk3hix
         u2(i) = 0.
 210  CONTINUE

      bad_y = bad_flag(ARG3)
      CALL STL2(xpts, arg_3, tol(3), nx, bad_y, bad_flag_result, 
     .   u3, v, w, nout3, IP)
      IF (nout3 .GT. res_hi_ss(X_AXIS)) THEN
         nout3 = nx
         DO 300 i = 1, nx
	    u3(i) = FLOAT(i)
 300     CONTINUE
      ENDIF

      DO 310 i = nout3+1, wrk4hix
         u3(i) = 0.
 310  CONTINUE

* Truncate result(j=1) to next lowest integer - its going to be
* the set of coordinate values to sample at.

      DO 400 i = 1, wrk6hix
	 w(i) = 0.
 400  CONTINUE

      CALL merge (u1,nout1,u2,nout2,w,nmerge1,wrk6hix)
      IF (nmerge1 .GT. res_hi_ss(X_AXIS)) THEN
        write (errtxt,*) 'nmerge1 too large', nmerge1
	 GOTO 999
      ENDIF

      DO 500 i = 1, wrk5hix
	 v(i) = 0.
 500  CONTINUE
      DO 510 i = nmerge1+1, wrk6hix
	 w(i) = 0.
 510  CONTINUE

      CALL merge (w,nmerge1,u3,nout3,v,nmerge2,wrk5hix)
      IF ((nmerge2 .EQ. res_hi_ss(X_AXIS)) .OR. 
     .    (nmerge2 .EQ. res_hi_ss(X_AXIS)+1) ) THEN
        nmerge2 = res_hi_ss(X_AXIS)  ! Returning all the data
      ELSE IF (nmerge2 .GT. res_hi_ss(X_AXIS)) THEN
	 GOTO 999
      ENDIF

      DO 600 i = 1, nmerge2
        result(i,k,k,k) = v(i)
 600  CONTINUE

      DO 700 i = nmerge2+1, res_hi_ss(X_AXIS)
        result(i,k,k,k) = bad_flag_result
 700  CONTINUE

      RETURN
 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN

      END
