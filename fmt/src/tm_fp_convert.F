	SUBROUTINE TM_FP_CONVERT(buffer,ncnt,author_cpu)

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
* convert VAX or IEEE big/little endian into the currently active CPU
* representation

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX/VMS, DEC Ultrix, and SUN
*
* revision 0.0 - 2/11/92 - based on code from Dave Borg-Breen
*  (ref. VAX Professional, August '91)
* bug fix *sh* 3/92 - need special check for exact zero
* replaced "elif" syntax with
*	else
*	   if
*  for SGI port	 - kob 4/8/92
* *kob* 10/96 	- Linux port.  Linux didnt like tabs 
*		  between "include"
*	          and what was being included.  removed them

* calling argument declarations:
	INTEGER ncnt, author_cpu
	REAL    buffer(ncnt)

* internal variable declarations:
        REAL*4    dum
        INTEGER*4 idum, k

#ifdef sun
      include 'tmap_format/xmachine_byte.cmn'
      BYTE bb(4),bdum
#else
#	if unix
                include 'tmap_format/xmachine_int1.cmn'
      		INTEGER*1 bb(4),bdum
#	else
      		INCLUDE 'TMAP_FORMAT:XMACHINE_BYTE.CMN'
      		BYTE bb(4),bdum
# 	endif
#endif

	EQUIVALENCE (bb(1),dum), (idum,dum)


***********   CONVERT between machine types   **************
*** use the pre-processor to select the target word type ***
#ifdef sun
* SUN is the platform in use
      IF     ( author_cpu .EQ. cptype_vax ) THEN   ! VAX -> SUN
	 DO 100 k = 1, ncnt
	    dum = buffer(k)
            IF ( idum .EQ. 0 ) GOTO 100 ! check for exact zero
	    bdum = bb(1)
	    bb(1) = bb(2)-1
	    bb(2) = bdum
	    bdum = bb(4)
	    bb(4) = bb(3)
	    bb(3) = bdum
	    buffer(k) = dum
 100     CONTINUE
      ELSEIF ( author_cpu .EQ. cptype_dec ) THEN   ! SUN <-> DECstation
         DO 200 k = 1, ncnt
	    dum = buffer(k)
            IF ( idum .EQ. 0 ) GOTO 200 ! check for exact zero
	    bdum = bb(1)
	    bb(1) = bb(4)
	    bb(4) = bdum
	    bdum = bb(2)
	    bb(2) = bb(3)
	    bb(3) = bdum
	    buffer(k) = dum
 200     CONTINUE
      ENDIF

#else
#if unix
* DECstation is the platform in use

      IF     ( author_cpu .EQ. cptype_vax ) THEN  ! VAX -> DECstation
	 DO 100 k = 1, ncnt
	    dum = buffer(k)
            IF ( idum .EQ. 0 ) GOTO 100 ! check for exact zero
	    bdum = bb(1)
	    bb(1) = bb(3)
	    bb(3) = bdum
	    bdum = bb(2)
	    bb(2) = bb(4)
	    bb(4) = bdum-1
	    buffer(k) = dum
 100     CONTINUE
      ELSEIF ( author_cpu .EQ. cptype_sun ) THEN  ! SUN <-> DECstation
	 DO 200 k = 1, ncnt
	    dum = buffer(k)
            IF ( idum .EQ. 0 ) GOTO 200 ! check for exact zero
	    bdum = bb(1)
	    bb(1) = bb(4)
	    bb(4) = bdum
	    bdum = bb(2)
	    bb(2) = bb(3)
	    bb(3) = bdum
	    buffer(k) = dum
 200     CONTINUE
      ENDIF

#else
* VAX is the platform in use
      IF     ( author_cpu .EQ. cptype_dec ) THEN    ! DECstation -> VAX
	 DO 100 k = 1, ncnt
	    dum = buffer(k)
            IF ( idum .EQ. 0 ) GOTO 100 ! check for exact zero
	    bdum = bb(1)
	    bb(1) = bb(3)
	    bb(3) = bdum
	    bdum = bb(4)
	    bb(4) = bb(2)
	    bb(2) = bdum+1
	    buffer(k) = dum
 100     CONTINUE
      ELSEIF ( author_cpu .EQ. cptype_sun ) THEN    ! SUN -> VAX
	 DO 200 k = 1, ncnt
	    dum = buffer(k)
            IF ( idum .EQ. 0 ) GOTO 200 ! check for exact zero
	    bdum = bb(2)
	    bb(2) = bb(1)+1
	    bb(1) = bdum
	    bdum = bb(4)
	    bb(4) = bb(3)
	    bb(3) = bdum
	    buffer(k) = dum
 200     CONTINUE
      ENDIF
#endif
#endif

      RETURN
      END
