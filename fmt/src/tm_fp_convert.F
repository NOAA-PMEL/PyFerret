	SUBROUTINE TM_FP_CONVERT(buffer,ncnt,author_cpu)

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
