	SUBROUTINE EQUAL_PERMUTE( string, perm, status )

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
* from a string of the form "perm=XYZT", return array perm with the ordering
* expressed as integerd

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   9/1/89
* V230: 4/92 (SUN port) - bug fix on loop: "fill in any missing dimensions"

* calling argument declarations
	INTEGER		perm(4), status
	CHARACTER*(*)	string

* internal variable declarations
	INTEGER	TM_LENSTR1, STR_UPCASE, equal_pos, slen, pos, vax_code,
     .		idim, ndim, i
	CHARACTER*1	ax

* internal parameter declarations:
	CHARACTER*1	tab
	PARAMETER     ( tab = CHAR(9) )

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* initialize
	slen	  = TM_LENSTR1( string )
	equal_pos = INDEX( string, '=' )
	IF ( equal_pos .EQ. 0 .OR. equal_pos .EQ. slen ) GOTO 5100

* skip blanks and tabs
	DO 100 pos = equal_pos+1, slen
	   IF ( string(pos:pos) .NE. ' '
     .	  .AND. string(pos:pos) .NE. tab ) GOTO 200
 100	CONTINUE
	GOTO 5100

* decode permutation string
 200	IF ( slen-pos .GT. 3 ) GOTO 5100
	ndim = 0
	DO 250 pos = pos, slen
	   vax_code = STR_UPCASE( ax, string(pos:pos) )
	   DO 210 idim = 1, 4
	      IF ( ax .EQ. ss_dim_name(idim)
     .	      .OR. ax .EQ. ww_dim_name(idim) ) THEN
	         ndim = ndim + 1
	         perm(ndim) = idim
	         GOTO 250
	      ENDIF
 210	   CONTINUE
	   GOTO 5100
 250	CONTINUE

* fill in any missing dimensions (e.g. "perm=YX" is interpreted as YXZT)
	DO 350 ndim = ndim+1, 4
	   DO 320 idim = 1, 4
	      DO 310 i = 1, ndim-1   ! loop of slots already set (4/92 bug fix)
 310	      IF ( perm(i) .EQ. idim ) GOTO 320
	      perm(ndim) = idim
	      GOTO 350
 320	   CONTINUE
 350	CONTINUE

* check that no axis is given twice
	DO 410 idim = 1, 4
	   DO 400 i = idim+1, 4
 400	   IF ( perm(i) .EQ. perm(idim) ) GOTO 5100
 410	CONTINUE

* success
	status = ferr_ok
	RETURN

* syntax error
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, string, *5000 )
	END
