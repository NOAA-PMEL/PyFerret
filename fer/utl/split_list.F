	SUBROUTINE SPLIT_LIST( mode, out_lun, next_line, line_len )

* depending on the state of mode_gui either write this string to the
* out_lun unit number or send it to the GUI

* Note that line_len=0 means this routine will determine the line length

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V320:  11/9/94
*	 12/29/94 - added "mode" argument
* V400:	  6/20/95 - bug fix: LIST/FILE output sent to "default_list_lun"
*			should go to disk file, not to screen

#ifdef unix
	include 'ferret_cmn/ferret.parm'
        include 'ferret_cmn/gui.parm'
	include 'ferret_cmn/xprog_state.cmn'
        include 'ferret_cmn/xgui.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
        INCLUDE 'FERRET_CMN:GUI.PARM'
        INCLUDE 'FERRET_CMN:XGUI.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling arguments
	INTEGER	      mode, out_lun, line_len
	CHARACTER*(*) next_line

*internal variable declarations:
        INTEGER slen
        PARAMETER  ( slen = 255 )
#ifdef sun
        BYTE      fhol(slen)	! c-type Hollerith string buffer
#else
        INTEGER*1 fhol(slen)	! c-type Hollerith string buffer
#endif
	INTEGER	TM_LENSTR1, llen

* determine line length if not supplied
	IF ( line_len .LE. 0 ) THEN
	   llen = TM_LENSTR1(next_line)
	ELSE
	   llen = line_len
	ENDIF

* output to the desired device
!	IF ( mode_gui ) THEN    ! 6/20 - kludge
	IF ( mode_gui .AND. out_lun.NE.default_list_lun ) THEN
	   CALL TM_FTOC_STRNG( next_line(:llen), fhol, slen )
	   CALL FERRET_LIST_IN_WINDOW( fhol, ftxout_append )
	ELSE
	   WRITE ( out_lun, '(A)' ) next_line(:llen)
	ENDIF

	RETURN
	END
