*  dist2.F
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
* Given two variables, return the distance from each point to the previous one
* in 2-space (lon/lat on the globe?)
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


      SUBROUTINE dist2_init(id)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 'distance along 2-d path' )

      CALL ef_set_num_args(id, 3)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id, IMPLIED_BY_ARGS, NORMAL, NORMAL, 
     .                                             NORMAL, NORMAL, NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'V1')
      CALL ef_set_arg_desc(id, arg, 'Variable 1 in X')
      CALL ef_set_axis_influence_6d(id, arg, YES, NO, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'V2')
      CALL ef_set_arg_desc(id, arg, 'Variable 2 in X')
      CALL ef_set_axis_influence_6d(id, arg, YES, NO, NO, NO, NO, NO)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'param')
      CALL ef_set_arg_desc(id, arg, 
     . '=0 for simple rect dist, 1 for cos(lat), 2 for law of cosines ')
      CALL ef_set_axis_influence_6d(id, arg, YES, NO, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*  In this subroutine we compute the result
* 
      SUBROUTINE dist2_compute(id, arg_1, arg_2, arg_3, result)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

* Function is intended only for variables in the x direction

      REAL arg_1(mem1lox:mem1hix)
      REAL arg_2(mem2lox:mem2hix)
      REAL arg_3(mem3lox:mem3hix)
      REAL result(memreslox:memreshix)

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

      INTEGER	i, i1, r1, r2
      REAL	radlat1, radlat2, radlon1, radlon2, d1, d2, 
     .		pi, pi180, rad, parm
      CHARACTER*100 errtxt
      
      pi = 3.1415926535897932384
      pi180 = pi/ 180.
      rad = 6370.e5/100.  ! radius of the earth in meters

      CALL ef_get_one_val(id, ARG3, parm) 

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      r1 = arg_hi_ss(X_AXIS, ARG1) - arg_lo_ss(X_AXIS, ARG1) + 1
      r2 = arg_hi_ss(X_AXIS, ARG1) - arg_lo_ss(X_AXIS, ARG1) + 1
      IF (r1.NE.r2) THEN
         errtxt = 'Both variables must be on the X axis and have the same length'
	 GOTO 999
      ENDIF

      i1 = arg_lo_ss(X_AXIS,ARG1)+1
      result(1) = bad_flag_result

* Distance on the globe (actual distance would be the result below, multiplied by
* earth radius)

      DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         radlon1 = bad_flag_result
         radlat1 = bad_flag_result

         IF ( arg_1(i1).NE.bad_flag(ARG1) .AND. arg_1(i1-1).NE.bad_flag(ARG1) ) THEN
            radlon1 =  pi180* arg_1(i1-1)
	    radlon2 =  pi180* arg_1(i1)
	 ENDIF
         IF ( arg_2(i1).NE.bad_flag(ARG2) .AND. arg_2(i1-1).NE.bad_flag(ARG2) ) THEN
            radlat1 =  pi180* arg_2(i1-1)
	    radlat2 =  pi180* arg_2(i1)
	 ENDIF

         IF ( radlat1.NE.bad_flag_result .AND. 
     .        radlon1.NE.bad_flag_result ) THEN

*  ref http://www.movable-type.co.uk/scripts/latlong.html

*  Linear distance, no cos correction
            IF (parm .EQ. 0) THEN
  	       result(i) = rad* (radlon1*radlon1 + radlat1*radlat1)**0.5

*  simple cosine-lat formula

            ELSE IF (parm .EQ. 1) THEN
	       d1 = (radlon2-radlon1) * cos( (radlat1+radlat2)/2. )
	       d2 = (radlat2-radlat1)
  	       result(i) = rad* (d1*d1 + d2*d2)**0.5

            ELSE

*  Great-circle distance, Law of cosines
* acos( sin(t1).sin(t2) + cos(t1).cos(t2).cos(s2-s1) )* R
* where  t is latitude, s is longitude, R is earth’s radius (mean radius = 6,371km)

	       result(i) = rad* acos( sin(radlat1)*sin(radlat2) + 
     .                        cos(radlat1)*cos(radlat2)*cos(radlon2-radlon1) )

            ENDIF

	 ELSE

	    result(i) = bad_flag_result

         ENDIF

	 i1 = i1 + arg_incr(X_AXIS,ARG1)
  100 CONTINUE

      RETURN

 9000 RETURN
 999  CALL EF_BAIL_OUT(id, errtxt)
      RETURN
      END
