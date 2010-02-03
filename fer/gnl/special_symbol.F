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
*       8/05 *acm*- Allow more digits in xrevision number
* V62   4/09 *acm*- Add new symbol DELTA_CPU. When it is evaluated,
*                   calls the intrinsic fcn Dtime and sets the value to that new
*                   delta-time since the last call.  Initialized in initizlize.F
* V62   4/09 *acm*- Add new symbol CLOCK_SECS. When it is evaluated, calls the
*                   intrinsic SYSTEM_CLOCK and sets the value to the delta-time 
*                   in seconds since Ferret was initialized.
* V62   4/09 *acm*- Add new symbols CURRENT_DATE, CURRENT_TIME. These get formatted
*                   date and time, unlike SESSION_DATE, SESSION_TIME which remain
*                   fixed at the start time of the Ferret session.
* V62   5/09 *acm*- make FERRET_MEMORY a special symbol.

* calling argument declarations:
	INTEGER	  isym, slen
	CHARACTER name*(*), value*(*)

* internal parameter declaration
	INTEGER nspecial
	PARAMETER (nspecial = 13)

* internal variable declarations:
	INTEGER STR_UPCASE, TM_LENSTR, i, i1, ierr, wsid, 
     .          nxpix, nypix, llen, icount, irate, imax
        REAL DTIME, TArray(2), dtime_res, clock_secs
	CHARACTER special(nspecial)*16, upname*16
	CHARACTER LEFINT*16, TM_FMT*12, NF_INQ_LIBVERS*80, NF_INQ_LIBVERS_*80
        CHARACTER*10 current_date, current_time, zone
	INTEGER itimes(8)

* names of special symbols (must be upper case)
	DATA special(1)/ 'PPL$XPIXEL' /,
     .	     special(2)/ 'PPL$YPIXEL' /,
     .	     special(3)/ 'BYTEORDER' /,
     .	     special(4)/ 'FERRET_VERSION' /,
     .	     special(5)/ 'FERRET_PLATFORM' /,
     .	     special(6)/ 'NETCDF_VERSION' /
     .	     special(7)/ 'FERRET_MEMORY' /,
     .	     special(8)/ 'SESSION_DATE' /,
     .	     special(9)/ 'SESSION_TIME' /,
     .	     special(10)/ 'DELTA_CPU' /
     .	     special(11)/ 'CLOCK_SECS' /
     .	     special(12)/ 'CURRENT_DATE' /
     .	     special(13)/ 'CURRENT_TIME' /

#ifdef unix
	include 'ferret.parm'
	include 'xplot_state.cmn'	! for PPL$XPIXEL
	include 'xrevision.cmn'
	include 'xtoday.cmn'
	include 'xplatform_type.cmn'
	include 'xvariables.cmn'
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

 200	GOTO (1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 
     .        1900, 2000, 2100, 2200, 2300) i

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
	value = TM_FMT(revision_level, 5, 12, llen)
	slen = llen
	RETURN

* PLATFORM_TYPE  *kob* 10/01
 1500	value = platform_type
	slen = TM_LENSTR(platform_type)
	RETURN

* NETCDF_VERSION
 1600	CONTINUE
#ifdef usingDODSf2cUnderscore
        value = NF_INQ_LIBVERS_()
#else
	value = NF_INQ_LIBVERS()
#endif
	slen = TM_LENSTR(value)
	RETURN

* FERRET_MEMORY
 1700	value = TM_FMT(max_mem_blks*mem_blk_size/1.E6,3,12,llen)
	slen = llen
	RETURN

* SESSION_DATE
 1800	value = today_date
	slen = 9
	RETURN

* SESSION_TIME
 1900	value = today_time
	slen = 5
	RETURN


* DELTA_CPU
 2000	dtime_res = DTIME(TArray)
	value = ' '
	value = TM_FMT(dtime_res, 6, 8, slen)
	RETURN

* CLOCK_SECS
 2100	CALL SYSTEM_CLOCK(icount, irate, imax)
        clock_secs = FLOAT(icount-clock_start_count)/FLOAT(irate)
	IF (clock_secs .LT. 0) THEN
           CALL WARN('Evaluating CLOCK_SECS, call to '//
     .     'SYSTEM_CLOCK. Count has exceeded its max and reset.')
           CALL WARN( 'This value of CLOCK_SECS is invalid. '//
     .     'Resetting to new initial value.')
           clock_secs = 0.
	   clock_start_count = FLOAT(icount)/FLOAT(irate)
	ENDIF
	value = ' '
	value = TM_FMT(clock_secs, 6, 8, slen)
	RETURN

* CURRENT_DATE
 2200	CONTINUE

#ifdef AIX_XLF
	CALL AIX_DATE( current_date )
#elif F90_DATE_TIME
	CALL FDATE(current_date)
#else
	CALL GET_DATE_AND_TIME (current_date, current_time)
#endif
        value = current_date
	slen = 9
	RETURN

* CURRENT_TIME
 2300	CONTINUE

#ifdef AIX_XLF
	CALL AIX_TIME( current_time )
#elif F90_DATE_TIME
	CALL FTIME(current_time)
#else
	CALL GET_DATE_AND_TIME (current_date, current_time)  ! doesnt get seconds

	CALL Date_and_Time(current_date, current_time, Zone, itimes)
        WRITE (current_time,1000) itimes(5), itimes(6), itimes(7)
 1000   FORMAT (2(I2.2,":"), I2.2)

#endif
	value = current_time
	slen = 8
	RETURN

	END
