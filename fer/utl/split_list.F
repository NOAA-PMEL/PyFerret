	SUBROUTINE SPLIT_LIST( mode, out_lun, next_line, line_len )

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
