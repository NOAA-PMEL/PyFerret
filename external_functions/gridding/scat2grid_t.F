*
*  scat2grid_t.F
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
* May 2003 
* 11-Jul-2003  Clean up section where we get box limits

* Input is a set of T coordinates on an abstract axis, and an output time axis.
* Returns a time series which is the number of points that fall inside of each T cell. 
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


      SUBROUTINE  scat2grid_t_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

************************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CHARACTER*126 buff
      WRITE (buff, 10)
   10 FORMAT ('Count number of t points in each cell of output axis')
      CALL ef_set_desc(id, buff)

      CALL ef_set_num_args(id, 2)
c      CALL ef_set_num_args(id, 3)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, 
     .      IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      CALL ef_set_num_work_arrays(id, 1)

* output grid is determined by argument 2. 

      arg = 1
      CALL ef_set_arg_name(id, arg, 'TPTS')
      CALL ef_set_arg_desc(id, arg, 
     .     'T coordinates of scattered input time coordinates')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'TAXIS')
      CALL ef_set_arg_desc(id, arg, 
     .     'Output time axis')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

c      arg = 3
c      CALL ef_set_arg_type(id, arg, STRING_ARG)
c      CALL ef_set_arg_name(id, arg, 'WHICH')
c      CALL ef_set_arg_desc(id, arg, 
c     .     'result to compute: SUM or AVG')
c      CALL ef_set_axis_influence(id, arg, NO, NO, NO, YES)

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
      SUBROUTINE scat2grid_t_work_size(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
* 
* Set the work arrays,  X/Y/Z/T dimensions
*
* ef_set_work_array_dims(id,array #,xlo,ylo,zlo,tlo,xhi,yhi,zhi,thi)
*
      INTEGER ntout, nt2
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)

      ntout = 1 + arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2)

      nt2 = (ntout+1)* 2

* tax  output t axis
      CALL ef_set_work_array_dims (id, 1, 1, 1, 1, 1, nt2, 1, 1, 1)

* workspace for computing average or variance; same as result T axis

c      CALL ef_set_work_array_dims (id, 2, 
c     .                             arg_lo_ss(T_AXIS,ARG2), 1, 1, 1, 
c     .                             arg_hi_ss(T_AXIS,ARG2), 1, 1, 1)

      RETURN
      END

*
* In this subroutine we compute the result
*

c      SUBROUTINE scat2grid_t_compute(id, arg_1, arg_2, arg_3,
c     .                               result, tax)

      SUBROUTINE scat2grid_t_compute(id, arg_1, arg_2,
     .                               result, tax)

*  arg_1  tpts  Scattered t values to be gridded.
*  arg_2  taxis to use for output

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id
      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)

c      REAL arg_3(mem3lox:mem3hix, mem3loy:mem3hiy, mem3loz:mem3hiz, 
c     .     mem3lot:mem3hit)


      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .     memresloz:memreshiz, memreslot:memreshit)

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
      INTEGER i, j, k, l, m, n
      INTEGER i1, i1n, i2, i2n, ntgrid
      INTEGER ntpts
      REAL tt, del, del1
      CHARACTER*250 errtxt

c      INTEGER STR_UPCASE, arg, iflag
c      INTEGER factr
c      CHARACTER*3 str, rtype

C  variables for checking axis characteristics (modulo axes)

      CHARACTER ax_name(4)*16, ax_units(4)*16
      LOGICAL backward(4), modulot(4), regular(4)

*  Dimension the work arrays

      REAL*8 tax(wrk1lox:wrk1hix/2, wrk1loy:wrk1hiy,
     .               wrk1loz:wrk1hiz, wrk1lot:wrk1hit)

      REAL*8 top_box

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)


* Get result type to compute 

c      arg = 3
c      CALL ef_get_arg_string(id, arg, str)
c      iflag = STR_UPCASE(rtype, str)

*  Check to see if output axis is modulo

      CALL ef_get_axis_info (id, 2, ax_name, ax_units, backward, 
     .                       modulot, regular)

*  Find number of points in scattered input points.  1-D arrays defining the 
*  scattered data points may lie on the X, Y, Z, or T axis of the input argument

      ntpts = 0

      DO 100 m = X_AXIS, T_AXIS
         IF (arg_lo_ss(m,ARG1) .GE. 1) THEN
            i1 = arg_lo_ss(m,ARG1)
            i1n = arg_hi_ss(m,ARG1)
            if (i1n-i1 .NE. 0) ntpts = 1 + (i1n - i1)
         ENDIF
 100  CONTINUE

      IF (ntpts .EQ. 0) GOTO 900

*  Check tax is a T axis

      i2 = arg_lo_ss(T_AXIS,ARG2)
      i2n = arg_hi_ss(T_AXIS,ARG2)
      ntgrid = i2n - i2 + 1

      IF (i2 .EQ. ef_unspecified_int4) THEN
         WRITE (errtxt, *) 'Second argument must be a T axis'
         GO TO 999
      ENDIF

C  Get coordinates of output axis.

      call ef_get_coordinates(id, ARG2, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG2), arg_hi_ss(T_AXIS, ARG2), tax)

C  Get cell boundaries

      CALL ef_get_box_lo_lim (id, ARG2, T_AXIS, 
     .   arg_lo_ss(T_AXIS, ARG2), arg_hi_ss(T_AXIS, ARG2), tax)

      CALL ef_get_box_hi_lim (id, ARG2, T_AXIS,
     .   arg_hi_ss(T_AXIS, ARG2), arg_hi_ss(T_AXIS, ARG2), top_box)

      L = res_hi_ss(T_AXIS)+1
      tax(L,1,1,1) = top_box

*  Compute result

*  Initialize counts

      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      DO 300 L = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
         result(i,j,k,l)= 0.
 300  CONTINUE

*  Loop over input tpts

      L = res_lo_ss(T_AXIS)

      DO 500 n = 1, ntpts

        CALL pickout (arg_1, n, tt)

*  If an output axis is modulo, apply modulo adjustment to that coordinate
*  of the scattered point.

        IF (modulot(4)) CALL modscat (tax, ntgrid, 1, tt)

        IF ( tt  .NE. bad_flag(ARG3) ) THEN
           DO 400 L = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 
              IF (n .EQ. 1 .AND. tt .EQ. tax(L,1,1,1) ) THEN
                 result(i,j,k,l) = result(i,j,k,l) + 1.
              ELSE IF (tt .GT. tax(L,1,1,1) .AND. 
     .                 tt .LE. tax(L+1,1,1,1) ) THEN
                result(i,j,k,l) = result(i,j,k,l) + 1.
              ENDIF
  400      CONTINUE

         ENDIF

 500  CONTINUE

c      IF (rtype .EQ. 'AVG') THEN
c         factr = 1./FLOAT(ntpts)
c         DO 600 L = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 
c            result(i,j,k,l) = result(i,j,k,l)* factr
c 600     CONTINUE
c      ENDIF

      RETURN

900   CONTINUE

      IF (ntpts .EQ. 0) THEN
        WRITE (errtxt, 30)
      ENDIF
      GOTO 999 

 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN

   30 FORMAT ('No data in scattered T points')

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
************************************************************************

      END

      SUBROUTINE pickout (atim, n, val)


*  ACM 5/2003
*  Called from the scat2grid_t function.
*  Pick out nth item from atim, which is really 4-D 
*  with normal axes on three of the four axes.

      REAL atim(*)
      REAL val
      INTEGER n

      val = atim(n)

      RETURN
      END
