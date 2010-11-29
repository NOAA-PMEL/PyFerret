	SUBROUTINE STORE_SYS_STRINGS( cmnd_text, cx, mr,
     .				      pos, nlines, status )

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


* Execute a shell command and save the resulting strings in the variable mr

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V530:  9/00 *sh*

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xvariables.cmn'
	include	'xcontext.cmn'
	include	'xrisc.cmn'

* calling argument declarations:
	INTEGER	cx, mr, pos, nlines, status
	CHARACTER*(*) cmnd_text

* internal variable declarations:
	INTEGER	TM_LENSTR1,clen, old_len, istat, new_mr
	REAL*8	 ADD_C_POINTER, sysout_ptr

* --- end of introductory code ---

* length of string to be stored
	clen = TM_LENSTR1(cmnd_text)

* convert command to a null-terminated C string
	CALL TM_FTOC_STRNG (cmnd_text,
     .			    risc_buff, size_rbuff )

* execute the shell command
	CALL GET_SYS_CMND(sysout_ptr, nlines, risc_buff, istat)
	IF (istat .NE. 0) THEN
	   CALL ERRMSG(ferr_insuff_memory, status,
     .			'dynamic memory for string', *5000)
	ELSE
	   status = ferr_ok
	ENDIF


* enlarge the context to accomodate the new strings
* ... It is OK to modify cx in this blase way since the Ferret syntax doesnt
*     support "{stuff}[cx mods]". cx mods can be applied to a system output
*     string list only via "LET var={stuff}" and then using "var[cx mods]"
	old_len = cx_hi_ss(cx, x_dim)
! note: the "2000" in this line matches the same in COUNT_NUMBER_LIST.F
	cx_hi_ss(cx, x_dim) = cx_hi_ss(cx, x_dim) + nlines - 2000
	cx_hi_ww(x_dim, cx) = cx_hi_ss(cx, x_dim)

* do we need to allocate a larger block of memory?
	IF ( cx_hi_ss(cx, x_dim) .GT. mr_nblks(mr)*mem_blk_size ) THEN
	   CALL CREATE_TEMP_MEM_VAR( cx, new_mr, status )
	   IF ( status .NE. ferr_ok ) GOTO 5000
	   mr_c_pointer(new_mr) = ADD_C_POINTER(mr_c_pointer(mr), 0,
     .				(mr_blk1(new_mr)-mr_blk1(mr))*mem_blk_size)
	   CALL XFER_C_PTRS( mr_c_pointer(mr),     1, 0,
     .			     mr_c_pointer(new_mr), 1, 0, old_len )
	   mr_type(mr) = ptype_float ! a trick so that C strings are not freed
	   CALL DELETE_VARIABLE(mr)  ! when DELETE_VARIABLE is called
	   mr = new_mr
	ELSE
	   mr_hi_ss(mr, x_dim) = cx_hi_ss(cx, x_dim)
	   mr_hi_ww(x_dim, mr) = cx_hi_ss(cx, x_dim)
 	ENDIF

* transfer the strings into this mr (and free the pointer)
	CALL XFER_C_PTRS(sysout_ptr,0,0,mr_c_pointer(mr),1,pos-1,nlines)
	CALL FREE_C_POINTER( sysout_ptr )

* success
	status = ferr_ok

 5000	RETURN
	END	
