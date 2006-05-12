*
* tax_units.F
*
* Jing Y. Li 
* May 4th 2006
*
* This function returns year specified by the first argument (variable 
* containing time values) from the second argument (variable from which time 
* encoding will be inferred).
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
* 2006-05-11 It only works with the unit nameing convention in xunits_data.F

      SUBROUTINE tax_units_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V

      CALL ef_set_desc(id,
     . 'Returns years of time axis coordinate values' )
   
*      CALL ef_set_num_work_arrays(id,1)

      CALL ef_set_num_args(id, 2)

      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)

      CALL ef_set_axis_reduction(id, REDUCED, REDUCED, REDUCED, REDUCED)

      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'A')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'this arg is passed through')
*      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES)
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'B')
      CALL ef_set_arg_unit(id, arg, ' ')

      CALL ef_set_arg_desc(id, arg, 'this arg is passed through')
      CALL ef_set_axis_influence(id, arg, NO, NO, NO, NO)

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE tax_units_compute(id, arg_1, arg_2, result)

*      INCLUDE 'tax.h' 
      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INCLUDE 'xunits.cmn_text'
      EXTERNAL xunits_data

      INTEGER iun
      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy,
     .           mem2loz:mem2hiz, mem2lot:mem2hit)

      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

      REAL*8 taxdat(wrk1lox:wrk1hix, wrk1loy:wrk1hiy,
     .              wrk1loz:wrk1hiz, wrk1lot:wrk1hit/2) 

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable's axes. 

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
*      CHARACTER*20 datebuf
      INTEGER iyear, day_of_mon
      CHARACTER*3 cmon
      INTEGER numtimes
      CHARACTER*16 axname(4), axunits(4)
      LOGICAL bkw2(4), mod2(4), reg2(4)

      INTEGER i,j,k,l
      INTEGER i2, j2, k2, l2

      CALL ef_get_res_subscripts(id, res_lo_ss, res_hi_ss, res_incr)
      CALL ef_get_arg_subscripts(id, arg_lo_ss, arg_hi_ss, arg_incr)
      CALL ef_get_bad_flags(id, bad_flag, bad_flag_result)

*     Get axis information
      CALL ef_get_axis_info(id, ARG2, axname, axunits, bkw2, mod2, reg2)

      i2 = arg_lo_ss(X_AXIS,ARG2)
      DO 400 i=res_lo_ss(X_AXIS), res_hi_ss(X_AXIS)

         j2 = arg_lo_ss(Y_AXIS,ARG2)
         DO 300 j=res_lo_ss(Y_AXIS), res_hi_ss(Y_AXIS)

           k2 = arg_lo_ss(Z_AXIS,ARG2)
           DO 200 k=res_lo_ss(Z_AXIS), res_hi_ss(Z_AXIS)

              l2 = arg_lo_ss(T_AXIS,ARG2)
              DO 100 l=res_lo_ss(T_AXIS), res_hi_ss(T_AXIS)

                DO 99 iun=pun_1st_time, pun_last_time 
*                  print *, un_name(iun)
                  CALL LOWER(axunits(4))
                  IF( un_name(iun) .EQ. axunits(4)) THEN
                    result(i,j,k,l) = un_convert(iun) 
*                    print *, un_name(iun) 
                  END IF
 99             CONTINUE

                l2 = l2 + arg_incr(T_AXIS,ARG2)
 100          CONTINUE

               k2 = k2 + arg_incr(Z_AXIS,ARG2)
 200        CONTINUE

            j2 = j2 + arg_incr(Y_AXIS,ARG2)
 300     CONTINUE

         i2 = i2 + arg_incr(X_AXIS,ARG2)
 400  CONTINUE


*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
  900 CONTINUE

      RETURN
      END

C 
C Subroutine file 'length.for' to facilitate string manipulation.
C
C Version: 5.20
C Date: 1998, March 6
C
C Coded by: Ludek Klimes
C     Department of Geophysics, Charles University Prague,
C     Ke Karlovu 3, 121 16 Praha 2, Czech Republic,
C     E-mail: klimes@seis.karlov.mff.cuni.cz
C
C.......................................................................
C
C This file consists of the following external procedures:
C     LOWER...Subroutine changing a given character string to lowercase.
C             LOWER
C     LENGTH..Integer function to determine the length of a string
C             without trailing blanks.
C             LENGTH
C
C=======================================================================
C
C     
C
      SUBROUTINE LOWER(TEXT)
      CHARACTER*(*) TEXT
C
C Subroutine changing a given character string to lowercase.
C
C Input:
C     TEXT... A given string.
C
C Output:
C     TEXT... The given string converted to lowercase.
C
C-----------------------------------------------------------------------
C
      EXTERNAL LENGTH
      INTEGER  LENGTH
C
C-----------------------------------------------------------------------
C
      CHARACTER*1 LETTER
      INTEGER ISHIFT,I
C
      ISHIFT=ICHAR('a')-ICHAR('A')
      DO 10 I=1,LENGTH(TEXT)
        LETTER=TEXT(I:I)
        IF('A'.LE.LETTER.AND.LETTER.LE.'Z') THEN
          TEXT(I:I)=CHAR(ICHAR(LETTER)+ISHIFT)
        END IF
   10 CONTINUE
      RETURN
      END
C
C=======================================================================
C
C     
C
      INTEGER FUNCTION LENGTH(TEXT)
      CHARACTER*(*) TEXT
C
C Subroutine to determine the length of a string without trailing
C blanks.
C
C Input:
C     TEXT... Character string.
C
C Output:
C     LENGTH..Length of the string without trailing blanks.
C             LENGTH=1 for a blank string.
C
C No subroutines and external functions required.
C
C Date: 1995, August 18
C Coded by Ludek Klimes
C
C-----------------------------------------------------------------------
C
      INTEGER I
C
C.......................................................................
C
      DO 1 I=LEN(TEXT),1,-1
        IF(TEXT(I:I).NE.' ') THEN
          GO TO 2
        END IF
    1 CONTINUE
      I=1
    2 CONTINUE
      LENGTH=I
C
      RETURN
      END
C
C=======================================================================
C                                                                 

