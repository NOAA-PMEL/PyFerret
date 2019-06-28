	SUBROUTINE SHOW_ABORTED_MEM_STATE(i8_rqst_size)

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
* This routine is called following a ferr_insuff_memory error to display
* diagnostic information about the state of memory at the time of the failure.
* Specifically, how much memory was tied up for what purposes

* V702 3/17 *sh*
* V751 *acm* 6/19 Issue 1903: memory set and reported in terms of bytes

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xinterrupt.cmn'

* calling argument declarations:
	INTEGER*8 i8_rqst_size

* local variable declarations:
	INTEGER UVAR_NUMBER, TM_LENSTR1, uvar, slen, slen2, slen3
	INTEGER*8 object_mem
	CHARACTER LEFINT*8, LEFINT8*16, EXPR_NAME*8, TM_FMT*48,
     .		arg_num*8, name_code*8, mem_str*16, mem_unit*2, percent*8
	REAL*8 rmax_mem

* initialize
	rmax_mem = max_mem_allowed  ! so percentages become floating point

* write introductory line
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'    At the moment that the memory limit was reached',0 )

* write "Argument 2 of the command was being evaluated:" expression
	IF (num_uvars_in_cmnd .GT. 1) THEN
	   arg_num = LEFINT(current_cmnd_uvar, slen)
	   name_code = EXPR_NAME(current_cmnd_uvar)
	   uvar = UVAR_NUMBER( name_code )
	   slen2 = TM_LENSTR1(uvar_text(uvar))
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'    argument '//arg_num(:slen)//' was being evaluated: '//
     .		uvar_text(uvar)(:slen2), 0)
	ENDIF

* summarize how memory was allocated
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'    memory was committed as follows:',0 )

	IF (permanent_mem .GT. 0) THEN
	   mem_str = LEFINT8(permanent_mem, slen2)
	   percent = TM_FMT (permanent_mem/rmax_mem*100, 0, 8, slen3)
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'        - to LOAD/PERM storage: '//
     .		mem_str(:slen2)//' ('//percent(:slen3)//'%)' ,0 )
	ENDIF

	IF (current_cmnd_uvar .GT. 1) THEN
	   arg_num = LEFINT(current_cmnd_uvar-1, slen)
	   mem_str = LEFINT8(completed_args_mem, slen2)
	   percent = TM_FMT (completed_args_mem/rmax_mem*100, 0, 8, slen3)
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'        - to arguments 1-'//arg_num(:slen)//
     .		': '//mem_str(:slen2)//' ('//percent(:slen3)//'%)' ,0 )
	ENDIF

	object_mem = essential_mem-completed_args_mem
	mem_str = LEFINT8(object_mem, slen2)
	percent = TM_FMT (object_mem/rmax_mem*100, 0, 8, slen3)
	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'        - to objects used in computation: '//
     .		mem_str(:slen2)//' ('//percent(:slen3)//'%)' ,0 )


* Report the size of the requested object
	mem_str = LEFINT8(i8_rqst_size, slen2)
	percent = TM_FMT (i8_rqst_size/rmax_mem*100, 0, 8, slen3)

	CALL MEM_TO_BYTES (i8_rqst_size, mem_str, slen2, mem_unit)
	IF (i8_rqst_size .EQ. 0) mem_unit = ' '

	CALL SPLIT_LIST(pttmode_explct, show_lun,
     .		'    The size of the requested object was: '//
     .		mem_str(:slen2)//' '//mem_unit//' ('//percent(:slen3)//'%)' ,0 )

* report the activity that was underway
* ... could optionally investigate the is_act states of isp and isp-1 to report
*  "reading dtaa while gathering for average" or whatever ...

	RETURN
	END
