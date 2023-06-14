*
* lon_lat_time_string.F
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
* Jan 2015
* 6-D function
* v75  *acm* 3/18  Issue 1856, changes to allow for timesteps in fractional seconds
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
*                        CUSTO   - user defined axis
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


      SUBROUTINE lon_lat_time_string_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     .        'Generate strings from longitude,latitude,time')
      CALL ef_set_num_args(id, 5)

      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_result_type(id, STRING_RETURN)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'longitude')
      CALL ef_set_arg_desc(id, arg, 'Station longitude')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'latitude')
      CALL ef_set_arg_desc(id, arg, 'Station latitude')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

      arg = 3
      CALL ef_set_arg_name(id, arg, 'time')
      CALL ef_set_arg_desc(id, arg, 'numeric times')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

      arg = 4
      CALL ef_set_arg_name(id, arg, 'time_axis')
      CALL ef_set_arg_desc(id, arg, 
     .     'Varible with a time axis to use to convert time to time strings')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

      arg = 5
      CALL ef_set_arg_name(id, arg, 'string to prepend')
      CALL ef_set_arg_type (id, arg, STRING_ARG)
      CALL ef_set_arg_desc(id, arg, 'A label to start the strings')
      CALL ef_set_axis_influence_6d(id, arg, NO, NO, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END


*
* In this subroutine we compute the result
*
      SUBROUTINE lon_lat_time_string_compute(id, arg_1, arg_2, arg_3, arg_4, 
     .          arg_5, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'


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
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)


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
      INTEGER TM_LENSTR
      INTEGER i,  j,  k,  l,  m,  n
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER slen, len1, len2, outlen, llen, get_prec, dlen
      
      CHARACTER*2   ns
      CHARACTER*48  TM_FMT, datestring, lonstr, latstr, ilab
      CHARACTER*256 lab, outstring
      REAL ddate, alon, alat

 1100  FORMAT(i1)
 1200  FORMAT(i2.2)
 1300  FORMAT(i3.3)

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

* return date string to precision of seconds
      get_prec = 6

      CALL ef_get_arg_string(id, ARG5, lab)
      slen = TM_LENSTR(lab)

      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 600 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
      m1 = arg_lo_ss(E_AXIS,ARG1)
      DO 500 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)
      l1 = arg_lo_ss(T_AXIS,ARG1)
      DO 400 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

         k1 = arg_lo_ss(Z_AXIS,ARG1)
         DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

          j1 = arg_lo_ss(Y_AXIS,ARG1)
          DO 200 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

           i1 = arg_lo_ss(X_AXIS,ARG1)
           DO 100 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

            IF ( arg_1(i1,j1,k1,l1,m1,n1) .EQ. bad_flag(ARG1) .OR.
     .           arg_2(i1,j1,k1,l1,m1,n1) .EQ. bad_flag(ARG2) .OR.
     .           arg_3(i1,j1,k1,l1,m1,n1) .EQ. bad_flag(ARG3) ) THEN

C     Get the string value of dates and put them in the result.

               outstring = '***'
	       CALL EF_PUT_STRING (outstring, 3, result(i,j,k,l,m,n))

            ELSE

	       alon = arg_1(i1,j1,k1,l1,m1,n1)
	       IF (alon .LT. 0) alon = 360 + alon
	       lonstr = TM_FMT(alon, 4, 5, len1)

	       ns = 'N_'
	       alat = arg_2(i1,j1,k1,l1,m1,n1)
	       IF (alat .LT. 0) THEN
	          alat = ABS(alat)
		  ns = 'S_'
	       ENDIF
	       latstr = TM_FMT(alat, 4, 5, len2)
	          
               ddate = arg_3(i1,j1,k1,l1,m1,n1)
               CALL EF_GET_AXIS_DATES (id, ARG4, ddate, T_AXIS, 1, get_prec, dlen, datestring)
               outstring = lab(:slen)//'_'//lonstr(:len1)//
     .           'E_'//latstr(:len2) //ns//datestring(1:11)

cc Just use a sequence number...
cc
cc	       IF (arg_hi_ss(X_AXIS,ARG1) .LT. 10) THEN
cc	          WRITE (ilab, 1100) i1
cc		  llen = 1
cc	       ELSEIF (arg_hi_ss(X_AXIS,ARG1) .LT. 100) THEN
cc	          WRITE (ilab, 1200) i1
cc		  llen = 2
cc	       ELSEIF (arg_hi_ss(X_AXIS,ARG1) .LT. 1000) THEN
cc	          WRITE (ilab, 1300) i1
cc		  llen = 3
cc	       ENDIF
cc               outstring = lab(:slen)//'_'//ilab(:llen)
     
	       outlen = TM_LENSTR(outstring)
	       CALL EF_PUT_STRING (outstring, outlen, result(i,j,k,l,m,n))

            ENDIF

            i1 = i1 + arg_incr(X_AXIS,ARG1)
 100       CONTINUE

           j1 = j1 + arg_incr(Y_AXIS,ARG1)
 200      CONTINUE

          k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300     CONTINUE

         l1 = l1 + arg_incr(T_AXIS,ARG1)
 400    CONTINUE
         m1 = m1 + arg_incr(E_AXIS,ARG1)
 500    CONTINUE
         n1 = n1 + arg_incr(F_AXIS,ARG1)
 600    CONTINUE


      RETURN

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN
      END
