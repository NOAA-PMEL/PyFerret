*
* zaxreplace_xyt.F
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
* august 2004
*
* V580:10/04 *acm* Fix for bug 1037. Before the loop calling hunt_r4,
*                  set the initial guess for nsrc_lo to 1 not 0.  
*
* Replace Z axis, using linear interpolation 
* (same as zaxreplace, as first step towards zaxreplace_zlev)
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


      SUBROUTINE zaxreplace_xyt_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 descr

      WRITE (descr, 20)
   10 FORMAT ('regrid V onto Z axis of ZAX based on ',
     .    'Z values in ZVALS. ZVALS may vary in XYT ')

      CALL ef_set_desc(id, descr )

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, YES, YES, YES, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'V')
      CALL ef_set_arg_desc(id, arg, 
     .        'Variable on native z axis')
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'THICKNESS')
      WRITE (descr, 20)
   20 FORMAT ('Thickness corresponding to input Z axis values ',
     .        'in units of output Z axis')

      CALL ef_set_arg_desc(id, arg, descr)
      CALL ef_set_axis_influence(id, arg, YES, YES, NO, YES)

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
* In this subroutine we compute the result
*
      SUBROUTINE zaxreplace_xyt_compute(id, arg_1, arg_2, arg_3, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

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
      INTEGER nsrc_lo, klo, khi
      REAL frac, zval

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* 
*  Loop over x,y,t of the input fields
*

      klo = arg_lo_ss(Z_AXIS, ARG2)
      khi = arg_hi_ss(Z_AXIS, ARG2)

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

* For each z line interpolate. 

               nsrc_lo = 0

               nsrc_lo = 1    ! fixes bug where if zval equals 1st point in axis
                              ! we got no match.

               DO 100 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

* ... next Z axis value (arg3) to locate

                  zval = arg_3(i3,j3,k,l3)

* ... search the Z field (arg_2) for the index just below this value
* ... note that nsrc_lo is always referenced to a starting index of 1

	          CALL HUNT_R4(arg_2(i2,j2,mem2loz,l2), 
     .                         1, khi, zval, nsrc_lo, frac)

* ... interpolate data arg_1 field to get result

                  IF ( frac .LT. 0.0 ) THEN
* ... no luck -- the Z axis value we seek is outside the range in com2
                     result(i,j,k,l) = bad_flag_result
                  ELSEIF ( arg_1(i1,j1,nsrc_lo,l1) .EQ. 
     .                            bad_flag(ARG1) ) THEN
                     result(i,j,k,l) = bad_flag_result
                  ELSEIF ( frac .EQ. 1.0 ) THEN                  ! exactly on point
	             result(i,j,k,l) = arg_1(i1,j1,nsrc_lo,l1)
                  ELSE
	             result(i,j,k,l) =  frac* arg_1(i1,j1,nsrc_lo,l1)
     .			     + (1-frac)* arg_1(i1,j1,nsrc_lo+1,l1)
               ENDIF
 100        CONTINUE


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


	SUBROUTINE HUNT_R4 (x1,lo_dim,hi_dim,x2,ndx_lo, frac)

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
*
*	Routine to determine nearest neighbor index (lower index)
*	on GRID1 for points on GRID2.

*	Taken from Numerical Recipes, Chapter 3, Section "Search With
*	Correlated Values"

* V450:  7/97 - new, cloned from double prec. version hunt.F
*             - also return "frac" - the interpolation factor for ndx_lo 

	integer		lo_dim,hi_dim,ndx_lo,ndx_mid,ndx_hi,inc

	real		x1(lo_dim:hi_dim),x2, frac

	logical		ascnd

********************************************************************************

*	TRUE IF ASCENDING GRID COORDINATES, FALSE OTHERWISE
	ascnd = x1(lo_dim+1) .gt. x1(lo_dim)
	
*	IF INITIAL GUESS IS NO GOOD, GO IMMEDIATELY TO BISECTION
	if (ndx_lo .lt. lo_dim .or. ndx_lo .gt. hi_dim) then
	   ndx_lo = lo_dim - 1
	   ndx_hi = hi_dim + 1
	   goto 3
	end if

	inc = 1

*	QUICKLY GET A GOOD ESTIMATE FOR LOCATION IN GRID1
	if (x2 .ge. x1(ndx_lo) .eqv. ascnd) then
*	   HUNT UP FOR PLACEMENT IN GRID
1	   ndx_hi = ndx_lo + inc
	   
	   if (ndx_hi .gt. hi_dim) then 
*	      HUNT OVER -- OFF HIGH END OF GRID
	      ndx_hi = hi_dim + 1
!	   else if (x2 .ge. x1(ndx_hi) .eqv. ascnd) then (ORIGINAL CODE)
	   else if (x2 .gt. x1(ndx_hi) .eqv. ascnd) then
	      ndx_lo = ndx_hi
	      inc    = inc + inc
	      goto 1
	   end if
	else
*	   HUNT DOWN FOR PLACEMENT
	   ndx_hi = ndx_lo
2	   ndx_lo = ndx_hi - inc

	   if (ndx_lo .lt. lo_dim) then
*	      FALL OFF LOW END OF GRID -- HUNT IS OVER
	      ndx_lo = lo_dim - 1
	   else if (x2 .lt. x1(ndx_lo) .eqv. ascnd) then
	      ndx_hi = ndx_lo
	      inc = inc + inc
	      goto 2
	   end if	
	end if

*	USE BISECTION TO DETERMINE EXACT LOCATION IN GRID
3	IF (ndx_hi - ndx_lo .eq. 1) THEN

* return result -- including interpolation fraction

	   IF (ndx_lo.GE.lo_dim .AND. ndx_lo.LT.hi_dim) THEN
	      IF (x2 .EQ. x1(ndx_lo)) THEN
	         frac = 1.0
	      ELSE
	         frac = (x1(ndx_hi)-x2) / (x1(ndx_hi)-x1(ndx_lo))
	      ENDIF
	   ELSE    
	      frac = -999.0
	   ENDIF
	   return
	ENDIF

	ndx_mid = (ndx_hi + ndx_lo)/2
	if (x2 .gt. x1(ndx_mid) .eqv. ascnd) then
	   ndx_lo = ndx_mid
	else
	   ndx_hi = ndx_mid
	end if

	goto 3


	end
