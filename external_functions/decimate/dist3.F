*  dist3.F
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
* Given three variables, return the distance from each point to the previous one
* in 3-space.
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


      SUBROUTINE dist3_init(id)

      IMPLICIT NONE
      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id, 'distance along 3-d path' )

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
      CALL ef_set_arg_name(id, arg, 'V3')
      CALL ef_set_arg_desc(id, arg, 'Variable 3 in X')
      CALL ef_set_axis_influence_6d(id, arg, YES, NO, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*  In this subroutine we compute the result
* 
      SUBROUTINE dist3_compute(id, arg_1, arg_2, arg_3, result)

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

      INTEGER	i, i1, r1, r2, r3
      REAL	d1, d2, d3
      CHARACTER*100 errtxt

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

      r1 = arg_hi_ss(X_AXIS, ARG1) - arg_lo_ss(X_AXIS, ARG1) + 1
      r2 = arg_hi_ss(X_AXIS, ARG1) - arg_lo_ss(X_AXIS, ARG1) + 1
      r3 = arg_hi_ss(X_AXIS, ARG1) - arg_lo_ss(X_AXIS, ARG1) + 1
      IF (r1.NE.r2 .OR. r1.NE.r3) THEN
         errtxt = 'All variables must be on the X axis and have the same length'
	 GOTO 999
      ENDIF

      i1 = arg_lo_ss(X_AXIS,ARG1)+1
      result(1) = bad_flag_result

      DO 100 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)
         d1 = bad_flag_result
         d2 = bad_flag_result
         d3 = bad_flag_result

         IF ( arg_1(i1).NE.bad_flag(ARG1) .AND. arg_1(i1-1).NE.bad_flag(ARG1) )
     .       d1 = arg_1(i1) - arg_1(i1-1)
         IF ( arg_2(i1).NE.bad_flag(ARG2) .AND. arg_2(i1-1).NE.bad_flag(ARG2) )
     .       d2 = arg_2(i1) - arg_2(i1-1)
         IF ( arg_3(i1).NE.bad_flag(ARG3) .AND. arg_3(i1-1).NE.bad_flag(ARG3) )
     .       d3 = arg_3(i1) - arg_3(i1-1)

         IF ( d1.NE.bad_flag_result .AND. 
     .        d2.NE.bad_flag_result .AND. 
     .        d3.NE.bad_flag_result) THEN

	    result(i) = (d1*d1+d2*d2+d3*d3)**0.5
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
