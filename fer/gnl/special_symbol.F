	SUBROUTINE SPECIAL_SYMBOL( isym, name, value, slen )

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
* this routine is 2-in-1 to handle "special" (internally defined) symbols
* it acts like an encapsulated special symbol "object" (FORTRAN ...)
* if isym=0 on input then this is a request to translate the passed "name"
*	if the given name is the name of a special symbol return its definition
*	in value and its length in slen. Else return slen=0
* if slen is a positive integer then return the name of that symbol
*	or a blank name if the integer is larger than the number of symbols 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V522 7/00 *sh* - extracted from symbol_command.F
* V530  8/00 *sh* - bug (feature) fix so that SHOW SYM doesn't eveluate
*			PPL$XPIXEL
*      12/00 *sh* - added symbols SESSION_DATE and SESSION_TIME
*      10/01 *kob*- add symbol FERRET_PLATFORM to display the platform
*                   the executable was built on. 

* calling argument declarations:
	INTEGER	  isym, slen
	CHARACTER name*(*), value*(*)

* internal parameter declaration
	INTEGER nspecial
	PARAMETER (nspecial = 7)

* internal variable declarations:
	INTEGER STR_UPCASE, i, ierr, wsid, nxpix, nypix, TM_LENSTR
	CHARACTER special(nspecial)*16, upname*16
	CHARACTER LEFINT*16

* names of special symbols (must be upper case)
	DATA special(1)/ 'PPL$XPIXEL' /,
     .	     special(2)/ 'PPL$YPIXEL' /,
     .	     special(3)/ 'BYTEORDER' /,
     .	     special(4)/ 'FERRET_VERSION' /,
     .	     special(5)/ 'FERRET_PLATFORM' /,
     .	     special(6)/ 'SESSION_DATE' /,
     .	     special(7)/ 'SESSION_TIME' /

#ifdef unix
	include 'ferret.parm'
	include 'xplot_state.cmn'	! for PPL$XPIXEL
	include 'xrevision.cmn'
	include 'xtoday.cmn'
	include 'xplatform_type.cmn'
* #ifdef for endianness info.
#ifdef sun
	include 'xmachine_byte.cmn'
#else
	include 'xmachine_int1.cmn'
#endif

#endif

* is this a query by integer or by name?
	IF (isym .GE. 1) THEN 
* ***** QUERY BY INTEGER ******
	   IF (isym.GT.nspecial) THEN
	      name = ' '
	      RETURN
	   ENDIF
	   name = special(isym)
	   RETURN
	ENDIF

* ***** QUERY BY NAME *****
* identify the given name
	ierr = STR_UPCASE( upname, name )
	DO 100 i = 1, nspecial
	   IF (upname .EQ. special(i)) GOTO 200
 100	CONTINUE

* not a recognized special symbol
	slen = 0
	RETURN

 200	GOTO (1100, 1200, 1300, 1400, 1500, 1600, 1700) i

* PPL$XPIXEL, PPL$YPIXEL
 1100	CONTINUE
* ... determine the current output window (GKS wkstn ID, that is)
 1200	DO 1230 wsid = 1, max_windows
           IF ( wn_active(wsid) ) GOTO 1240
 1230   CONTINUE
* ... no currently active window
	value = '0'
	slen = 1
	RETURN
 1240	CALL GET_WINDOW_PIXELS(wsid, nxpix, nypix)
	IF ( i.EQ.1 ) THEN
	   value = LEFINT(nxpix,slen)
	ELSE
	   value = LEFINT(nypix,slen)
	ENDIF
	RETURN

* BYTEORDER - get byteorder of this architecture
 1300	IF (active_cpu .EQ. cptype_sun) THEN
	   value = 'BIG'
	   slen = 3
	ELSE
	   value = 'LITTLE'
	   slen = 6
	ENDIF
	RETURN

* FERRET_VERSION
 1400	value = ' '
	WRITE (value,'(F5.2)') revision_level
	slen = 5
	RETURN

* PLATFORM_TYPE  *kob* 10/01
 1500	value = platform_type
	slen = TM_LENSTR(platform_type)
	RETURN

* SESSION_DATE
 1600	value = today_date
	slen = 9
	RETURN

* SESSION_TIME
 1700	value = today_time
	slen = 5
	RETURN

	END
