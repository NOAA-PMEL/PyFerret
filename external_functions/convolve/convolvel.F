*
* convolvel.F
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
** Ansley Manke
* Feb 1999
*
* This external function convolves the component grid, com, with the weight 
* function, wt  along the L axis (see Ferret routine CONVOLVE)
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


      SUBROUTINE convolvel_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      CHARACTER*100 fcn_desc
      WRITE (fcn_desc, 10)
   10 FORMAT ('Convolve L component of variable with weight function')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'COM')
      CALL ef_set_arg_desc(id, arg, 
     .  'Variable in T (and perhaps X,Y,Z) to convolve')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'WEIGHT')
      CALL ef_set_arg_desc(id, arg, 'Weight function')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)
*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
	SUBROUTINE convolvel_compute (id, arg_1, arg_2, result )

*
*
*  From FERRET subroutine CONVOLVE, for the i-axis.
*
* convolve the component grid, com, with the weight function, wt
* along axis idim
* note: the component context may not be of adequate size for the full
*	calculation.  Missing data flags will be inserted where computation is
*	impossible
* also: when bad data points are encountered in the component data all
*	result data depending on it are flagged as bad, too


      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id
      REAL bad_flag(EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .     mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .     mem2lot:mem2hit)
     .     
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .      memresloz:memreshiz, memreslot:memreshit)

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
      REAL comp, sum, weight
      INTEGER i, j, k, l
      INTEGER xlen, ylen, zlen, tlen, ll
      INTEGER i1, j1, k1, l1
      INTEGER i2, j2, k2, l2
      INTEGER wlen, hlen

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)
      
* CONVOLVE ALONG T AXIS (arg_1) using weights (arg_2)

* Half the weighting function; weights from -hlen to hlen.

      xlen = (arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1) 
      ylen = (arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1) 
      zlen = (arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2) + 1) 
      tlen = (arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1) 

      wlen = max(xlen, ylen, zlen, tlen)
      IF  (MOD(wlen,2) .EQ. 0) wlen = wlen + 1
      hlen = wlen/ 2

      i1 = arg_lo_ss(X_AXIS,ARG1)
      DO 500 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j1 = arg_lo_ss(Y_AXIS,ARG1)
         DO 400 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 300 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)
       
               l1 = arg_lo_ss(T_AXIS,ARG1)
               DO 200 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS) 

                  sum = 0.0
                  i2 = arg_lo_ss(X_AXIS,ARG2)
                  j2 = arg_lo_ss(Y_AXIS,ARG2)
                  k2 = arg_lo_ss(Z_AXIS,ARG2)
                  l2 = arg_lo_ss(T_AXIS,ARG2)

                  DO 100 ll = -hlen, hlen

                     IF (l1+ll .LT. arg_lo_ss(T_AXIS,ARG1)  .OR.
     .                   l1+ll .GT. arg_hi_ss(T_AXIS,ARG1) ) THEN
                        result(i,j,k,l) = bad_flag_result
                        GOTO 190

                     ELSE
                        comp = arg_1(i1,j1,k1,l1+ll)
                        IF (i2 .LT. arg_lo_ss(X_AXIS,ARG2)  .OR.
     .                      i2 .GT. arg_hi_ss(X_AXIS,ARG2)  .OR.
     .                      j2 .LT. arg_lo_ss(Y_AXIS,ARG2)  .OR.
     .                      j2 .GT. arg_hi_ss(X_AXIS,ARG2)  .OR.
     .                      k2 .LT. arg_lo_ss(Z_AXIS,ARG2)  .OR.
     .                      k2 .GT. arg_hi_ss(Z_AXIS,ARG2)  .OR.
     .                      l2 .LT. arg_lo_ss(T_AXIS,ARG2)  .OR.
     .                      l2 .GT. arg_hi_ss(T_AXIS,ARG2)) THEN
                               weight = 0.
                        ELSE
                           weight = arg_2(i2,j2,k2,l2)
                        ENDIF

                        IF ( comp .EQ. bad_flag(ARG1)) THEN
                           result(i,j,k,l) = bad_flag_result
                           GOTO 190
                        ELSE
                           sum = sum + comp* weight
                        ENDIF
                     ENDIF

                     i2 = i2 + arg_incr(X_AXIS,ARG2)
                     j2 = j2 + arg_incr(Y_AXIS,ARG2)
                     k2 = k2 + arg_incr(Z_AXIS,ARG2)
                     l2 = l2 + arg_incr(T_AXIS,ARG2)

 100                 CONTINUE
                     result(i,j,k,l) = sum

 190                 l1 = l1 + arg_incr(T_AXIS,ARG1)
 200              CONTINUE

                  k1 = k1 + arg_incr(Z_AXIS,ARG1)
 300           CONTINUE

            j1 = j1 + arg_incr(Y_AXIS,ARG1)
 400     CONTINUE

         i1 = i1 + arg_incr(X_AXIS,ARG1)
 500  CONTINUE

      RETURN
      END	
