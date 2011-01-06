* chdir_pmel.F
*
* Change current working directory.
* -atw 10/17/01

      SUBROUTINE chdir_pmel_init(id)

      INCLUDE 'ferret_cmn/EF_Util.cmn'

      INTEGER id, arg

      CALL ef_version_test(ef_version)

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CALL ef_set_desc(id,'Change current working directory.' )
      CALL ef_set_num_args(id, 1)
      CALL ef_set_axis_inheritance(id, IMPLIED_BY_ARGS, 
     .     IMPLIED_BY_ARGS, IMPLIED_BY_ARGS, IMPLIED_BY_ARGS)
      CALL ef_set_piecemeal_ok(id, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_type(id, arg, STRING_ARG)
      CALL ef_set_arg_name(id, arg, 'TO_DIRECTORY')
      CALL ef_set_arg_desc(id, arg, 'desired working directory')
      CALL ef_set_axis_influence(id, arg, YES, YES, YES, YES) 

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

*
* In this subroutine we compute the result
*
      SUBROUTINE chdir_pmel_compute(id, arg_1, result)

      INCLUDE 'ferret_cmn/EF_Util.cmn'
      INCLUDE 'ferret_cmn/EF_mem_subsc.cmn'

      INTEGER id

      REAL bad_flag(1:EF_MAX_ARGS), bad_flag_result
      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, 
     .           mem1loz:mem1hiz, mem1lot:mem1hit)
      REAL result(memreslox:memreshix, memresloy:memreshiy,
     .            memresloz:memreshiz, memreslot:memreshit)

      INTEGER res_lo_ss(4), res_hi_ss(4), res_incr(4)
      INTEGER arg_lo_ss(4,1:EF_MAX_ARGS), arg_hi_ss(4,1:EF_MAX_ARGS),
     .     arg_incr(4,1:EF_MAX_ARGS)

      INTEGER itest, ilen
      CHARACTER arg1_text*160
      INTEGER CHDIR, TM_LENSTR1

* **********************************************************************
*                                            USER CONFIGURABLE PORTION |
*                                                                      |
*                                                                      V
      CALL ef_get_arg_string(id, 1, arg1_text)
      ilen = TM_LENSTR1(arg1_text)

      WRITE(*,*) 'Changing directory to: ',arg1_text(1:ilen)
      itest = CHDIR(arg1_text(1:ilen))

*                                                                      ^
*                                                                      |
*                                            USER CONFIGURABLE PORTION |
* **********************************************************************

      RETURN 
      END

      INTEGER FUNCTION TM_LENSTR1 (line)

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
* determine the length of a character string as the position of the last 
* non-blank character
* unlike TM_LENSTR this routine will return a length of 1 (vs. 0) if the string
* is all blanks

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 05/05/87 - based on TM_LENSTR rev. 0.00
*
* calling argument declarations:
	CHARACTER*(*) line

* internal variable declarations:
	INTEGER	i, mright

* initialize: find highest possible right hand limit of string
	mright = LEN(line)

	DO 100 i = mright,1,-1
	IF (line(i:i) .NE. ' ') GOTO 200
 100	CONTINUE

* for all blanks
	TM_LENSTR1 = 1
	RETURN

* found non-blank
 200	TM_LENSTR1 = i
	RETURN
	END
