	SUBROUTINE EQUAL_PERMUTE( string, perm, status )

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
