	SUBROUTINE PARSE_COMMA_LIST ( buffer, istart, iend, max_items, 
     .				      num_it, it_start, it_end, status )

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
*
* find the boundaries of the "items" that are separated by commas
* istart and iend delimit the search region within the buffer and are
* both assumed to represent the positions of non-blank characters
* likewise the returned item limits will represent non-blank characters
*
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/7/86
* revision 1.0 - 7/28/88 - complete re-write to cope wit imbedded "[" and ""
* revision 1.1 - 8/9/88  - allow multiple commas for PLOT+ compatibility
* V230:  8/4/92 - error message lacks blanks
* V320: 12/29/94 - use WARN to avoid direct tty output (for GUI)
* Linux Port - 12/96 *kob
*	     - had to close open quotes in above comment line - linux f77
*	       was freaking out because of it...
* V530: *sh* 3/01 - remove surrounding quotes from items in list
* V532: *sh* 5/01 - ignore commas embedded in parentheses
*		  - allow for nested parens and brackets
* V533: *sh* 7/01 - allow both single and double quoted strings

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'	! for ttout_lun
	include 'xrisc.cmn'

* calling argument declarations:
	CHARACTER*(*)	buffer
	INTEGER		istart, iend, max_items, num_it,
     .			it_start(max_items), it_end(max_items), status

* internal variable declarations:
	INTEGER		ip, i, nest_level
	CHARACTER*8	LEFINT

* initialize
	risc_buff = buffer		! for RISC/FORTRAN
	num_it = 0
	ip = istart

* TOP OF LOOP
 10	CONTINUE

	IF ( buffer(ip:ip) .EQ. ' '
     .  .OR. buffer(ip:ip) .EQ. '	' ) THEN
* skip past leading tabs and blanks to find first character
	   ip = ip + 1
	   IF ( ip .GT. iend ) GOTO 1000
	   GOTO 10
	ELSE
* found first character of item
	   IF ( num_it .EQ. max_items ) GOTO 6000
	   num_it = num_it + 1
	   it_start(num_it) = ip
	ENDIF

* comma alone representing a place holder ?
	IF ( buffer(ip:ip) .EQ. ',' ) THEN
	   it_end(num_it) = ip
	   ip = ip + 1
	   IF ( ip .GE. iend ) GOTO 1000
	   GOTO 10
	ENDIF

* search for end of item
	ip = ip - 1
 30	ip = ip + 1
	IF ( ip .GE. iend ) THEN
	   it_end(num_it) = iend
	   GOTO 1000
	ENDIF

* ... ignore commas sandwiched between [ ]
	IF ( buffer(ip:ip) .EQ. '[' ) THEN
	   nest_level = -1  ! will be immediately incremented
	   DO 40 i = ip, iend
	      IF ( buffer(i:i) .EQ. '[' ) THEN
	         nest_level = nest_level + 1
	      ELSEIF ( buffer(i:i) .EQ. ']' ) THEN
	         IF (nest_level .GT. 0) THEN
	            nest_level = nest_level - 1
	         ELSE
	            ip = i
	            GOTO 30
	         ENDIF
	      ENDIF
 40	   CONTINUE
	   GOTO 5020	! unpaired

* ... ignore commas sandwiched between ( )
	ELSEIF ( buffer(ip:ip) .EQ. '(' ) THEN
	   nest_level = -1  ! will be immediately incremented
	   DO 45 i = ip, iend
	      IF ( buffer(i:i) .EQ. '(' ) THEN
	         nest_level = nest_level + 1
	      ELSEIF ( buffer(i:i) .EQ. ')' ) THEN
	         IF (nest_level .GT. 0) THEN
	            nest_level = nest_level - 1
	         ELSE
	            ip = i
	            GOTO 30
	         ENDIF
	      ENDIF
 45	   CONTINUE
	   GOTO 5020	! unpaired

* ... ignore commas sandwiched between " "
	ELSEIF ( buffer(ip:ip) .EQ. '"' ) THEN
	   DO 50 i = ip+1, iend
	      IF ( buffer(i:i) .EQ. '"' ) THEN
	         ip = i
	         GOTO 30
	      ENDIF
 50	   CONTINUE
	   GOTO 5020	! unpaired

* ... ignore commas sandwiched between ' '   ! 7/01
	ELSEIF ( buffer(ip:ip) .EQ. "'" ) THEN
	   DO i = ip+1, iend
	      IF ( buffer(i:i) .EQ. "'" ) THEN
	         ip = i
	         GOTO 30
	      ENDIF
	   ENDDO
	   GOTO 5020	! unpaired

* ... comma as terminator
	ELSEIF ( buffer(ip:ip) .EQ. ',' ) THEN
* ... search backwards for the last non-blank/tab
	   DO 60 i = ip-1,it_start(num_it),-1
	      IF ( buffer(i:i) .NE. ' '
     .	     .AND. buffer(i:i) .NE. '	' ) THEN
	         it_end(num_it) = i
	         ip = ip + 1
	         GOTO 10
	      ENDIF
 60	   CONTINUE
	   it_end(num_it) = it_start(num_it)
	   ip = it_start(num_it) + 1
	   GOTO 10

* any other character
	ELSE
	   GOTO 30

	ENDIF	   

* remove surrounding quotes from individual items
 1000	DO i = 1, num_it
	   IF (buffer(it_start(i):it_start(i)) .EQ. '"'
     .   .AND. buffer(it_end  (i):it_end  (i)) .EQ. '"' ) THEN
	      IF ( it_end(i) .GT. it_start(i)+1 ) THEN
	         it_start(i) = it_start(i) + 1
	         it_end(i)   = it_end(i)   - 1
	      ENDIF
	   ENDIF
	ENDDO
	RETURN

* error exits
 5000	RETURN
 5020	CALL ERRMSG( ferr_syntax, status,
     .		'unpaired quotations, parens, or brackets '
     .           //pCR//risc_buff(istart:iend), *5000 )

* warning exit
 6000	CALL WARN( 'command incompletely processed: '
     .		//buffer(istart:iend) )
	CALL WARN( 'Maximum items in command: '//LEFINT(max_items,i))
	RETURN

	END
