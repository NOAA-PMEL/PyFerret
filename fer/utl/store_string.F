	SUBROUTINE STORE_STRING( text, mr, offset, status )

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


* Store a given FORTRAN character string into dynamic memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V530:  8/00 *sh*

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xvariables.cmn'

* calling argument declarations:
	INTEGER	mr, offset, status
!	REAL    memory( mem_blk_size, max_mem_blks )
	CHARACTER*(*) text

* internal variable declarations:
	INTEGER	clen, istat

* --- end of introductory code ---

* length of string to be stored
	clen = LEN(text)

* convert to a null-terminated C string
! instead we are using the assumption that FORTRAN character arrays are
! passed with the correct starting address
!	CALL GET_WORK_SPC( 1+clen/4, wrk1, status )
!	IF (status .NE. ferr_ok) RETURN

!	CALL TM_FTOC_STRNG (text,
!     .			    memory(1,wrk1), clen )

* allocate dynamic storage and save the pointer in the Ferret grid
	CALL SAVE_C_STRING(text, clen,
     .			   mr_c_pointer(mr), offset, istat )
	IF (istat .NE. 0) THEN
	   CALL ERRMSG(ferr_insuff_memory, status,
     .			'dynamic memory for string', *5000)
	ELSE
	   status = ferr_ok
	ENDIF

 5000	RETURN
	END	
