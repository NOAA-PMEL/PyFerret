*
* convolvem.F
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
* May 2012
*
*
* This external function convolves the component grid, com, with the weight 
* function, wt  along the M axis (see Ferret routine CONVOLVE)
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


      SUBROUTINE convolvem_init(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V

      CHARACTER*100 fcn_desc

      WRITE (fcn_desc, 10)
   10 FORMAT ('Convolve M component of variable with weight function')
      CALL ef_set_desc(id, fcn_desc)

      CALL ef_set_num_args(id, 2)
      CALL ef_set_has_vari_args(id, NO)
      CALL ef_set_axis_inheritance_6d(id,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS,
     .                                IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'COM')
      CALL ef_set_arg_desc(id, arg, 
     .  'Variable in X (and perhaps Y,Z,T,E,F) to convolve')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              YES, YES, YES, YES, YES, YES)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'WEIGHT')
      CALL ef_set_arg_desc(id, arg, 'Weight function')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

*                                                                     ^
*                                                                     |
*                                           USER CONFIGURABLE PORTION |
***********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE convolvem_compute (id, arg_1, arg_2, result )

*
*
*  From FERRET subroutine CONVOLVE, for the i-axis.
*
* convolve the component grid, com, with the weight function, wt
* along axis idim
* note: the component context may not be of adequate size for the full
*       calculation.  Missing data flags will be inserted where computation is
*       impossible
* also: when bad data points are encountered in the component data all
*       result data depending on it are flagged as bad, too

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'

      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

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

***********************************************************************
*                                           USER CONFIGURABLE PORTION |
*                                                                     |
*                                                                     V
      REAL comp, sum, weight
      REAL xlen, ylen, zlen, tlen, elen, flen
      INTEGER i, j, k, l, m, n
      INTEGER hlo, hhi, mm, wlen
      INTEGER i1, j1, k1, l1, m1, n1
      INTEGER i2, j2, k2, l2, m2, n2


      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts_6d(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)
      
* CONVOLVE ALONG X AXIS (arg_1) using weights (arg_2)

* Half the weighting function; weights from -hlen to hlen (now: hlo to hhi)

      xlen = (arg_hi_ss(X_AXIS,ARG2) - arg_lo_ss(X_AXIS,ARG2) + 1) 
      ylen = (arg_hi_ss(Y_AXIS,ARG2) - arg_lo_ss(Y_AXIS,ARG2) + 1) 
      zlen = (arg_hi_ss(Z_AXIS,ARG2) - arg_lo_ss(Z_AXIS,ARG2) + 1) 
      tlen = (arg_hi_ss(T_AXIS,ARG2) - arg_lo_ss(T_AXIS,ARG2) + 1) 
      elen = (arg_hi_ss(E_AXIS,ARG2) - arg_lo_ss(E_AXIS,ARG2) + 1) 
      flen = (arg_hi_ss(F_AXIS,ARG2) - arg_lo_ss(F_AXIS,ARG2) + 1) 

      wlen = MAX(xlen, ylen, zlen, tlen, elen, flen)
      hhi = wlen / 2
      hlo = (-1) * hhi
      IF ( MOD(wlen,2) .EQ. 0 ) hlo = hlo + 1

      n1 = arg_lo_ss(F_AXIS,ARG1)
      DO 700 n = res_lo_ss(F_AXIS), res_hi_ss(F_AXIS)
       
         l1 = arg_lo_ss(T_AXIS,ARG1)
         DO 500 l = res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)
       
            k1 = arg_lo_ss(Z_AXIS,ARG1)
            DO 400 k = res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

               j1 = arg_lo_ss(Y_AXIS,ARG1)
               DO 300 j = res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

                  i1 = arg_lo_ss(X_AXIS,ARG1)
                  DO 200 i = res_lo_ss(X_AXIS), res_hi_ss(X_AXIS) 
       
                     m1 = arg_lo_ss(E_AXIS,ARG1)
                     DO 600 m = res_lo_ss(E_AXIS), res_hi_ss(E_AXIS)

                        sum = 0.0
                        i2 = arg_lo_ss(X_AXIS,ARG2)
                        j2 = arg_lo_ss(Y_AXIS,ARG2)
                        k2 = arg_lo_ss(Z_AXIS,ARG2)
                        l2 = arg_lo_ss(T_AXIS,ARG2)
                        m2 = arg_lo_ss(E_AXIS,ARG2)
                        n2 = arg_lo_ss(F_AXIS,ARG2)

                        DO 100 mm = hlo, hhi

                           IF (m1+mm .LT. arg_lo_ss(E_AXIS,ARG1)  .OR.
     .                         m1+mm .GT. arg_hi_ss(E_AXIS,ARG1) ) THEN
                              result(i,j,k,l,m,n) = bad_flag_result
                              GOTO 190
                           ELSE
                              comp = arg_1(i1,j1,k1,l1,m1+mm,n1)
                              IF ( comp .EQ. bad_flag(ARG1) ) THEN
                                 result(i,j,k,l,m,n) = bad_flag_result
                                 GOTO 190                  
                              ENDIF
                              weight = arg_2(i2,j2,k2,l2,m2,n2)
                              sum = sum + comp * weight
                           ENDIF

                           i2 = i2 + arg_incr(X_AXIS,ARG2)
                           j2 = j2 + arg_incr(Y_AXIS,ARG2)
                           k2 = k2 + arg_incr(Z_AXIS,ARG2)
                           l2 = l2 + arg_incr(T_AXIS,ARG2)
                           m2 = m2 + arg_incr(E_AXIS,ARG2)
                           n2 = n2 + arg_incr(F_AXIS,ARG2)

 100                    CONTINUE
                        result(i,j,k,l,m,n) = sum

 190                 m1 = m1 + arg_incr(E_AXIS,ARG1)
 600                 CONTINUE

	          i1 = i1 + arg_incr(X_AXIS,ARG1)
 200              CONTINUE

                  j1 = j1 + arg_incr(Y_AXIS,ARG1)
 300           CONTINUE

               k1 = k1 + arg_incr(Z_AXIS,ARG1)
 400        CONTINUE

            l1 = l1 + arg_incr(T_AXIS,ARG1)
 500     CONTINUE

         n1 = n1 + arg_incr(F_AXIS,ARG1)
 700  CONTINUE

      RETURN
      END
