	SUBROUTINE SHOW_1_UVAR( lun, uvar, line, lead_in, full )

* output some lines describing a user variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V402: 7/95 - extracted from XEQ_SHOW with additions for LET/D=

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xrisc.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN_TEXT'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	LOGICAL	full
	INTEGER	lun, uvar
	CHARACTER*(*) lead_in	! characters at margin - at least 1 character
	CHARACTER*(*) line	! line buffer -- 255 char
	
* local variable declarations:
	INTEGER TM_LENSTR1, slen, llen
	CHARACTER FULL_UVAR_NAME*64

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
	ENDIF

 4020   FORMAT (T20,'"',A,'"')

	RETURN
	END


