	SUBROUTINE SHOW_1_UVAR( lun, uvar, line, lead_in, full )

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
* output some lines describing a user variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V402: 7/95 - extracted from XEQ_SHOW with additions for LET/D=
* V510: 5/00 *sh* - show bad data flag if manually set
* v552: 4/03 *acm*  Increase length of variable names to 128 characters
*                   (variables are up to 128, so FULL_UVAR_NAME needs extra 
*                   space i.e. for a transformation specifier)

	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret.parm'
	include	'xvariables.cmn'
	include	'xrisc.cmn'

* calling argument declarations:
	LOGICAL	full
	INTEGER	lun, uvar
	CHARACTER*(*) lead_in	! characters at margin - at least 1 character
	CHARACTER*(*) line	! line buffer -- 255 char
	
* local variable declarations:
	INTEGER TM_LENSTR1, slen, llen
	CHARACTER FULL_UVAR_NAME*150, TM_FMT*12

* initialize
	IF (uvar_num_items(uvar) .EQ. uvar_deleted  ) RETURN
	llen = LEN(lead_in)

* variable name and definition
	risc_buff = lead_in//FULL_UVAR_NAME(uvar,slen)
	slen = slen + llen
	risc_buff = risc_buff(:slen)//' = '//uvar_text(uvar)		
	CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0)
	risc_buff = ' '

* extra information for "full" output
	IF ( full ) THEN
* ... title and units as "...title...(units)"
	   line = uvar_title(uvar)
	   IF ( line .EQ. ' ' ) THEN
	      slen = 3
	   ELSE
	      slen = 1
	   ENDIF
	   IF ( uvar_units(uvar) .NE. ' ' ) THEN
	      line = line(:TM_LENSTR1(line))//
     .                      ' ('//uvar_units(uvar)
	      line = line(:TM_LENSTR1(line))//')'
	   ENDIF
	   IF ( line .NE. ' ' ) THEN
	      WRITE (risc_buff,4020)
     .                       line(slen:TM_LENSTR1(line))
	      CALL SPLIT_LIST(pttmode_explct, lun,
     .                       risc_buff, 0)
	      risc_buff = ' '
	   ENDIF
* ... bad data flag if was manually set
	   IF ( uvar_bad_data(uvar) .NE. bad_val4 ) THEN
	      line = '          bad value flag = '//
     .			TM_FMT(uvar_bad_data(uvar),7,12,slen)
	      CALL SPLIT_LIST(pttmode_explct, lun,
     .                       line, 0)
	      risc_buff = ' '
	   ENDIF
	ENDIF

 4020   FORMAT (T10,'"',A,'"')

	RETURN
	END


