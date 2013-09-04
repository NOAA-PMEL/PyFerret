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
     . 'Return indices to sample from piecewise linear interpolation' )

      CALL ef_set_num_args(id, 7) 
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id, ABSTRACT, NORMAL, NORMAL, 
     .                                      NORMAL, NORMAL,   NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)
      CALL ef_set_num_work_arrays(id, 6)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'V1')
      CALL ef_set_arg_desc(id, arg, '1st variable to sample')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'V2')
      CALL ef_set_arg_desc(id, arg, '2nd variable to sample')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'V3')
      CALL ef_set_arg_desc(id, arg, '3rd variable to sample')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'Control parameter')
      CALL ef_set_arg_desc(id, arg,
     .          'Control parameter (e.g. 6 for ntol=rest=cont=1)')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'Tol1')
      CALL ef_set_arg_desc(id, arg, 'Tolerance for V1')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 6
      CALL ef_set_arg_name(id, arg, 'Tol2')
      CALL ef_set_arg_desc(id, arg, 'Tolerance for V2')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 7
      CALL ef_set_arg_name(id, arg, 'Tol3')
      CALL ef_set_arg_desc(id, arg, 'Tolerance for V3')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)



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
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

*
*     Use utility functions to get context information about the arguments.
*

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

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
* ef_set_work_array_dims_6d(id, array #,
*                           xlo, ylo, zlo, tlo, elo, flo,
*                           xhi, yhi, zhi, thi, ehi, fhi)
*

      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)
      INTEGER npts, iwork

      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)

      npts = 1 + arg_hi_ss(X_AXIS,ARG1) - arg_lo_ss(X_AXIS,ARG1)
  
* input x-axis coordinates
      iwork = 1
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                             npts, 1, 1, 1, 1, 1)

* internal computation, u1 for y1
      iwork = 2
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                             npts, 1, 1, 1, 1, 1)

* internal computation, u2 for y2
      iwork = 3
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                             npts, 1, 1, 1, 1, 1)

* internal computation, u3 for y3
      iwork = 4
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                             npts, 1, 1, 1, 1, 1)

* internal computation, v
      iwork = 5
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                             npts, 1, 1, 1, 1, 1)

* internal computation, w
      iwork = 6
      CALL ef_set_work_array_dims_6d(id, iwork,
     .                                1, 1, 1, 1, 1, 1,
     .                             npts, 1, 1, 1, 1, 1)
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
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)
      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
     .           mem3lot:mem3hit, mem3loe:mem3hie, mem3lof:mem3hif)
      REAL arg_4(mem4lox:mem4hix, mem4loy:mem4hiy, mem4loz:mem4hiz, 
     .           mem4lot:mem4hit, mem4loe:mem4hie, mem4lof:mem4hif)
      REAL arg_5(mem5lox:mem5hix, mem5loy:mem5hiy, mem5loz:mem5hiz, 
     .           mem5lot:mem5hit, mem5loe:mem5hie, mem5lof:mem5hif)
      REAL arg_6(mem6lox:mem6hix, mem6loy:mem6hiy, mem6loz:mem6hiz, 
     .           mem6lot:mem6hit, mem6loe:mem6hie, mem6lof:mem6hif)
      REAL arg_7(mem7lox:mem7hix, mem7loy:mem7hiy, mem7loz:mem7hiz, 
     .           mem7lot:mem7hit, mem7loe:mem7hie, mem7lof:mem7hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

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

      INTEGER i,  k
      INTEGER nx, nout1, nout2, nout3, nmerge1, nmerge2, ip
      REAL val, tol(3), bad_y
      CHARACTER*100 errtxt

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

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


* STL2 modified to truncate result to next lowest integer

      bad_y = bad_flag(ARG1)
      CALL STL2(xpts, arg_1, tol(1), nx, bad_y, bad_flag_result, 
     .   u1, v, w, nout1, IP)

      bad_y = bad_flag(ARG2)
      CALL STL2(xpts, arg_2, tol(2), nx, bad_y, bad_flag_result, 
     .   u2, v, w, nout2, IP)

      bad_y = bad_flag(ARG3)
      CALL STL2(xpts, arg_3, tol(3), nx, bad_y, bad_flag_result, 
     .   u3, v, w, nout3, IP)

* Truncate result(j=1) to next lowest integer - its going to be
* the set of coordinate values to sample at.

      DO 100 i = 1, nout1+nout2+1
	 w(i) = 0.
 100  CONTINUE

      CALL merge (u1,nout1,u2,nout2,w,nmerge1)

      DO 200 i = 1, res_hi_ss(1)
	 result(i,k,k,k,k,k) = 0.
 200  CONTINUE

      CALL merge (w,nmerge1,u3,nout3,result(1,k,k,k,k,k),nmerge2)

      DO 300 i = nmerge2+1, res_hi_ss(X_AXIS)
	 result(i,k,k,k,k,k) = bad_flag_result
 300  CONTINUE

      arg_7(1,k,k,k,k,k)  = nmerge2

      RETURN
 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN

      END

      SUBROUTINE STL2(X, Y, E, M, bad_y, bad_out, U, V, W, K, IP)

C PIECEWISE LINEAR APPROXIMATIONS OF FEWEST
C LINE SEGMENTS WITHIN GIVEN TOLERANCES.

C X,Y,E AND M CONTAIN INPUT DATA.
C U,V,K AND POSSIBLY W CONTAIN OUTPUT.

C IP IS A PARAMETER DETERMINING THE OPERATION
C  OF THE PROGRAM.

C X AND Y ARE INPUT DATA ARRAYS OF M ELEMENTS
C  X(I),Y(I) CONTAINS THE ITH DATA POINT.

C E MAY BE A SINGLE TOLERANCE OR A TABLE OF
C  TOLERANCES DEPENDING ON THE VALUE OF IP.
C IF E IS AN ARRAY, THEN E(I) IS THE TOLERANCE
C  ASSOCIATED WITH X(I),Y(I) AND E MUST CONTAIN
C  M NONNEGATIVE ELEMENTS.

C U AND V ARE OUTPUT ARRAYS OF K+1 ELEMENTS.
C  U IS A PARTITION OF THE INTERVAL (X(1),X(N))
C  WITH U(1)=X(1) AND U(K+1)=X(N).
C  V(I) IS AN ORDINATE TO BE ASSOCIATED WITH
C  U(I) IN THE APPROXIMATION.  (IF A CONTINUOUS
C  APPROXIMATION IS REQUESTED, THEN V(I) IS
C  'THE' ORDINATE TO BE ASSOCIATED WITH U(I).)

C IF A CONTINUOUS APPROXIMATION IS REQUESTED,
C  THEN W IS NOT USED.  IN THIS CASE THE ITH
C  APPROXIMATING SEGMENT IS THE STRAIGHT LINE
C  FROM U(I),V(I) TO U(I+1),V(I+1).

C IF A CONTINUOUS APPROXIMATION IS NOT
C  REQUESTED, THEN W IS A K-ELEMENT OUTPUT
C  ARRAY.  IN THIS CASE THE ITH APPROXIMATING
C  SEGMENT IS THE STRAIGHT LINE FROM
C  U(I),W(I) TO U(I+1),V(I+1), AND V(1) IS
C  SET EQUAL TO W(1).

C K IS THE NUMBER OF SEGMENTS IN THE PIECE-
C  WISE LINEAR APPROXIMATION GENERATED.  IN
C  CASE OF AN ERROR RETURN, K WILL BE SET TO
C  ZERO.

C THE CONTROL PARAMETER IP IS THE PRODUCT
C  OF THREE INDICATORS I1,I2 AND I3.
C  I1 INDICATES WHETHER OR NOT E IS AN
C     ARRAY OF TOLERANCES.
C     I1 = -1 INDICATES E IS AN ARRAY
C     I1 = +1 INDICATES E IS A SINGLE NUMBER.
C  I2 INDICATES WHETHER OR NOT THE
C     APPROXIMATION IS TO BE RESTRICTED TO
C     THE 'TOLERANCE BAND' ABOUT THE DATA.
C     I2 = 1 INDICATES NO BAND RESTRICTION
C     I2 = 2 INDICATES APPLY THIS RESTRICTION
C     (THE 'TOLERANCE BAND' IS A PIECEWISE
C     LINEAR BAND CENTERED AT THE DATA WHOSE
C     WIDTH IS DETERMINED BY THE TOLERANCES
C     AT THE DATA POINTS.)
C  I3 INDICATES WHETHER OR NOT THE
C     APPROXIMATION MUST BE CONTINUOUS.
C     I3 = 1 INDICATES CONTINUITY NOT REQUIRED
C     I3 = 3 INDICATES CONTINUITY IS REQUIRED

C CALL STL2 (X,Y,E,M,X,Y,E,M,IP) WILL NOT
C  CAUSE PROBLEMS PROVIDED THAT
C  EITHER A CONTINUOUS APPROXIMATION IS
C  REQUESTED, OR E IS A SUFFICIENTLY LARGE
C  ARRAY.

C THE PROGRAM PERFORMS THE FOLLOWING DATA
C  CHECKS.  ARE THE X-VALUES IN INCREASING
C  ORDER.  ARE THE TOLERANCE(S) NONNEGATIVE.
C  IS THE NUMBER OF DATA POINTS GREATER THAN
C  ONE.  IF ANY CHECK FAILS, THE PROGRAM
C  RETURNS WITH K SET EQUAL TO 0.  IN THIS
C  CASE NO FURTHER PROCESSING IS ATTEMPTED.

      IMPLICIT NONE
* Calling arguments
      REAL X(*), Y(*), E(*), U(*), V(*), W(*), bad_y, bad_out
      INTEGER M, K, IP

* Local declarations
      LOGICAL skipbad
      INTEGER N, ITCH, I, J, L, KEEP, INIT, INDC, IGRAZE,
     .        IDIOT, IT, IPIV, KP
      REAL EPSLN, SGN, XEYE, YEYE, TEMP1, SMIN, DX, 
     .     SVX, SVY, SVMN, TEMP2, XINIT, YINIT, SLOPE, SMAX, 
     .     SVMX, TEST

      N = M
      ITCH = IP
      J = 1
C ERROR CHECKS
      IF (N.LE.1) GO TO 400
      IF (E(1).LT.0.0) GO TO 400
      DO 10 L=2,N
        IF (X(L-1).GE.X(L)) GO TO 400
        IF (ITCH.GE.0) GO TO 10
        IF (E(L).LT.0.0) GO TO 400
   10 CONTINUE
C INITIALIZATION FOR ENTIRE PROGRAM
      EPSLN = E(1)
      SGN = 1.0
      KEEP = 1
      I = 1

      CALL PIECEWISE3_CHECKGAP (Y, bad_y, x, u, i, j, n, skipbad)
      IF (skipbad) GOTO 30

      U(i) = X(i)
      J = 2
      INIT = 1
      INDC = 0
      GO TO 30
C INITIALIZATION FOR EACH SEGMENT
   20 CONTINUE
      J = J + 1
      INIT = I
      INDC = 0
      IF (IABS(ITCH).LT.3) KEEP = I
      IF (IABS(IABS(ITCH)-4).NE.2) GO TO 30
C RESTRICTED TO TOLERANCE BAND
      XEYE = U(J-1)
      YEYE = V(J-1)
      TEMP1 = EPSLN
      IF (ITCH.LT.0) TEMP1 = TEMP1 + (SGN*E(I-1)-EPSLN)*(X(I)-U(J-1)
     * )/(X(I)-X(I-1))
      YINIT = YEYE - TEMP1 - TEMP1
      GO TO 40
   30 CONTINUE
C NOT RESTRICTED TO TOLERANCE BAND
      XEYE = X(I)
      YEYE = Y(I) + EPSLN
      YINIT = Y(I) - EPSLN
      IF (IABS(ITCH).EQ.1 .OR. I.EQ.1) GO TO 40
      TEMP1 = EPSLN
      IF (ITCH.LT.0) TEMP1 = SGN*E(I+1)
      SMIN = (Y(I+1)-YEYE-TEMP1)/(X(I+1)-XEYE)
      IF (ITCH.LT.0) TEMP1 = SGN*E(I-1)
      SMAX = (YEYE-Y(I-1)+TEMP1)/(XEYE-X(I-1))

      IF (KEEP.EQ.I-1) GO TO 50
      IT = I - 2
      XINIT = XEYE
      IPIV = I
      IGRAZE = I
      I = I + 1

      CALL PIECEWISE3_CHECKGAP (Y, bad_y, x, u, i, j, n, skipbad)
      IF (skipbad) GOTO 30

      GO TO 150
   40 CONTINUE
      IF (XEYE.GE.X(I)) I = I + 1
      IF (ITCH.LT.0) EPSLN = SGN*E(I)
      DX = X(I) - XEYE
      SMAX = (Y(I)+EPSLN-YEYE)/DX
      SMIN = (Y(I)-EPSLN-YEYE)/DX
   50 CONTINUE
      XINIT = XEYE
      IPIV = I
      IGRAZE = I
C DETERMINATION OF INDIVIDUAL SEGMENT
   60 CONTINUE
      IF (I.EQ.N) GO TO 260
      I = I + 1
      
C New i, so does it point to a bad-val?

      CALL PIECEWISE3_CHECKGAP (Y, bad_y, x, u, i, j, n, skipbad)
      IF (skipbad) GOTO 30

   70 CONTINUE
 
C TEST FOR NEW *MAX* SLOPE
      DX = X(I) - XEYE
      IF (ITCH.LT.0) EPSLN = SGN*E(I)
      TEMP1 = (Y(I)+EPSLN-YEYE)/DX
      TEST = TEMP1 - SMAX
      IF (SGN.LE.0.0) TEST = -TEST
      IF (TEST) 80, 90, 100
   80 CONTINUE
C TEST FOR END OF CANDIDATE SEGMENT
      TEST = TEMP1 - SMIN
      IF (SGN.LE.0.0) TEST = -TEST
      IF (TEST.LT.0.0) GO TO 210
      SMAX = TEMP1
   90 CONTINUE
C TEST FOR NEW *MIN* SLOPE
      IPIV = I
  100 CONTINUE
      TEMP2 = (Y(I)-EPSLN-YEYE)/DX
      TEST = TEMP2 - SMAX
      IF (SGN.LE.0.0) TEST = -TEST
      IF (TEST) 110, 120, 140
  110 CONTINUE
      TEST = SMIN - TEMP2
      IF (SGN.LE.0.0) TEST = -TEST
      IF (TEST) 120, 130, 60
  120 CONTINUE
      SMIN = TEMP2
  130 CONTINUE
      IGRAZE = I
      GO TO 60
C CHECK FOR PIVOT AT NEW EYE POINT
  140 CONTINUE
      IF (XEYE.EQ.X(IPIV)) GO TO 220
      IF (ITCH.LT.0) EPSLN = SGN*E(IPIV)
      INDC = 1
      SVX = XEYE
      SVY = YEYE
      SVMN = SMIN
      SVMX = SMAX
      XEYE = X(IPIV)
      YEYE = Y(IPIV) + EPSLN
      SMIN = SMAX
      SMAX = (YINIT-YEYE)/(XINIT-XEYE)
      IF (KEEP.GE.IPIV) GO TO 170
      IT = IPIV - 1
  150 CONTINUE
      TEMP2 = YEYE + EPSLN
      DO 160 L=KEEP,IT
        IF (ITCH.LT.0) TEMP2 = YEYE + SGN*E(L)
        TEMP1 = (Y(L)-TEMP2)/(X(L)-XEYE)
        TEST = TEMP1 - SMAX
        IF (SGN.LE.0.0) TEST = -TEST
        IF (TEST.LT.0.0) SMAX = TEMP1
  160 CONTINUE
  170 CONTINUE
      IF (IPIV.GE.I-1) GO TO 70
      IT = I - 2
      TEMP2 = YEYE - EPSLN
      IDIOT = IPIV
      DO 200 L=IDIOT,IT
        DX = X(L+1) - XEYE
        IF (ITCH.LT.0) TEMP2 = YEYE - SGN*E(L+1)
        TEMP1 = (Y(L+1)-TEMP2)/DX
        TEST = TEMP1 - SMAX
        IF (SGN.LE.0.0) TEST = -TEST
        IF (TEST) 180, 190, 200
  180   CONTINUE
        SMAX = TEMP1
  190   CONTINUE
        IPIV = L + 1
  200 CONTINUE
      GO TO 70
C END OF CURRENT SEGMENT
  210 CONTINUE
      TEMP2 = SMIN
      IF (I.EQ.N) GO TO 240
      KEEP = IGRAZE
      GO TO 250
  220 CONTINUE
      TEMP2 = SMAX
      IF (I.EQ.N) GO TO 230
      SGN = -SGN
      EPSLN = -EPSLN
      KEEP = IPIV
      GO TO 250
  230 CONTINUE
      IF (INDC.EQ.0 .OR. XEYE.NE.X(N-1)) GO TO 240
      XEYE = SVX
      YEYE = SVY
      SMIN = SVMN
      SMAX = SVMX
  240 CONTINUE
      U(J) = X(N-1)
      YINIT = Y(N-1)
      GO TO 270
  250 CONTINUE
      IF (IABS(IABS(ITCH)-4).NE.2) GO TO 300
C DETERMINE KNOT ON EDGE OF TOLERANCE BAND
      TEMP1 = 0.0
      IF (ITCH.LT.0) TEMP1 = EPSLN - SGN*E(I-1)
      TEMP1 = (Y(I)-Y(I-1)+TEMP1)/(X(I)-X(I-1))
      
      U(J) = (Y(I)+EPSLN-YEYE-TEMP1*X(I)+TEMP2*XEYE)/(TEMP2-TEMP1)

      GO TO 310
  260 CONTINUE
      U(J) = X(N)
      YINIT = Y(N)
  270 CONTINUE
C CONTINUITY CHECK FOR LAST SEGMENT
      IF (IABS(ITCH).GE.3 .OR. INIT.EQ.1) GO TO 290
      IT = INIT - 1
      SVMX = SMAX + SGN
      TEMP2 = YEYE + EPSLN
      DO 280 L=KP,IT
        IF (ITCH.LT.0) TEMP2 = YEYE + SGN*E(L)
        TEMP1 = (Y(L)-TEMP2)/(X(L)-XEYE)
        TEST = TEMP1 - SVMX
        IF (SGN.LE.0.0) TEST = -TEST
        IF (TEST.LT.0.0) SVMX = TEMP1
  280 CONTINUE
      IF (ABS(SVMX-SMAX+SVMX-SMIN).LE.ABS(SMAX-SMIN)) SMAX = SVMX
  290 CONTINUE
C NEARNESS CHECK FOR LAST SEGMENT
      TEMP2 = SMAX
      TEMP1 = YEYE + SMAX*(U(J)-XEYE)
      TEST = YINIT - TEMP1
      IF (SGN.LT.0.0) TEST = -TEST
      IF (TEST.GT.0.0) GO TO 310
      TEMP2 = SMIN
      TEMP1 = YEYE + SMIN*(U(J)-XEYE)
      TEST = YINIT - TEMP1
      IF (SGN.LT.0.0) TEST = -TEST
      IF (TEST.LT.0.0) GO TO 310
      TEMP2 = (YINIT-YEYE)/(U(J)-XEYE)
      V(J) = YINIT
      GO TO 320
  300 CONTINUE
      IF (IABS(ITCH).GE.3) GO TO 330
      U(J) = 0.5*(X(I)+X(I-1))
  310 CONTINUE
      V(J) = YEYE + TEMP2*(U(J)-XEYE)
  320 CONTINUE
      IF (XEYE.NE.XINIT) GO TO 330
      IF (IABS(ITCH).EQ.2) GO TO 360
      IF (IABS(ITCH).NE.6) GO TO 330
      IF (J.LE.2) GO TO 380
      GO TO 390
  330 CONTINUE
C RECOMPUTATION OF KNOT FOR CONTINUITY
      IF (J.LE.2) GO TO 370
      IF (SLOPE.EQ.TEMP2) GO TO 360
      YINIT = V(J-2)
      IF (IABS(ITCH).LT.3) YINIT = W(J-2)
      TEMP1 = (XEYE*TEMP2-U(J-2)*SLOPE+YINIT-YEYE)/(TEMP2-SLOPE)
      IF (IABS(ITCH).GE.3) GO TO 350
      IF (TEMP1.GT.XINIT) GO TO 360
      TEST = ABS(EPSLN)
      IDIOT = INIT - KP
      DO 340 L=1,IDIOT
        IT = INIT - L
        IF (TEMP1.GE.X(IT)) GO TO 350
        DX = Y(IT) - YEYE - TEMP2*(X(IT)-XEYE)
        IF (ITCH.LT.0) TEST = E(IT)
        IF (ABS(DX).GT.TEST) GO TO 360
  340 CONTINUE
  350 CONTINUE
      U(J-1) = TEMP1
      V(J-1) = YEYE + TEMP2*(U(J-1)-XEYE)
      IF (IABS(ITCH).LT.3) W(J-1) = V(J-1)
      GO TO 390
  360 CONTINUE
      W(J-1) = YEYE + TEMP2*(U(J-1)-XEYE)
      GO TO 390
  370 CONTINUE
      IF (IABS(ITCH).LT.3) GO TO 360
  380 CONTINUE
      V(1) = YEYE + TEMP2*(U(1)-XEYE)
  390 CONTINUE
      SLOPE = TEMP2
      KP = KEEP
      IF (I.LT.N) GO TO 20
      IF (X(N).EQ.U(J)) GO TO 400
      IF (IABS(ITCH).LT.3) W(J) = V(J)
      J = J + 1
      U(J) = X(N)
      V(J) = Y(N)
  400 CONTINUE
      IF (J.GE.2 .AND. IABS(ITCH).LT.3) V(1) = W(1)
      K = J

* Truncate to integer values: indices at which to sample

      DO 500 I = 1, K
         u(i) = INT(u(i))
  500 CONTINUE

      RETURN
      END

      SUBROUTINE PIECEWISE3_CHECKGAP (Y, bad_y, x, u, i, j, 
     .   n, skipbad)
      REAL Y(*), bad_y, x(*), u(*)
      INTEGER i, j, n
      LOGICAL skipbad

      skipbad = .FALSE.
	 
      IF ( Y(i).EQ.bad_y ) THEN
	 IF (i.GT.2) THEN
	    u(j) = x(i-1)
	       
	    IF ( j .gt. 1 .and. INT(u(j)) .LT. INT(u(j-1)) ) then
	       j = j - 1
	       DO WHILE (INT(u(j)) .LE. INT(u(j-1)) )
		     j = j - 1
	       ENDDO	  
	       u(j) = x(i-1)
	    ENDIF
            IF ( j .gt. 1 .and. INT(u(j)) .EQ. INT(u(j-1)) ) j = j - 1

            j = j + 1
	 ENDIF
         DO WHILE ( Y(i).EQ.bad_y .AND. i.LT.n)
            i = i + 1
         ENDDO
         
         IF (i.GT.1) u(j) = x(i-1)
         IF (i.GE.n) THEN
	    i = n
	    u(j) = x(n)
	 ENDIF
         IF ( j .gt. 1 .and. INT(u(j)) .EQ. INT(u(j-1)) ) j = j - 1

         j = j + 1
         u(j) = x(i)
         IF ( j .gt. 1 .and. INT(u(j)) .EQ. INT(u(j-1)) ) j = j - 1

         j = j + 1
         skipbad = .TRUE.
      ENDIF

      RETURN
      END


      subroutine merge(a,na,b,nb,c,nout)
c http://en.wikibooks.org/wiki/Algorithm_Implementation/Sorting/Merge_sort#Fortran
! Normal usage: NA+NB = NC
! B overlays C(NA+1:NC)

! modified to return nout instead of assuming treating NC, length of C as known.

      integer  NA,NB,NC         
      REAL  A(*), B(*), C(*)
 
      integer I,J,K
 
      I = 1
      J = 1
      K = 1
      do while(I .LE. NA .and. J .LE. NB)
         if (A(I) .LE. B(J)) then
            C(K) = A(I)
            I = I+1
         else
            C(K) = B(J)
            J = J+1
         endif
         K = K + 1
      enddo
      do while (I .LE. NA)
         C(K) = A(I)
         I = I + 1
          K = K + 1
      enddo
      do while (J .LE. NB)
         C(K) = B(J)
         J = J+1
         K = K + 1
      enddo

* remove duplicates

      j = 1
      do i = 2, k
         IF (c(i) .GT. c(i-1)) then
	    j = j + 1
	    c(j) = c(i)
	 endif
      enddo
      nout = j
      return
 
      end ! subroutine merge
