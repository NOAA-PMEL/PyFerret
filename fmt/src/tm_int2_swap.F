	SUBROUTINE TM_INT2_SWAP(ival2, author_cpu)

* byte-swap a 2-byte quantity to achieve big/little endian conversion

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX/VMS, DEC Ultrix, and SUN
*
* revision 0.0 - 2/13/92
* replaced "elif" syntax with
*	else
*	   if
*  for SGI port	 - kob 4/8/92
* bug fix -4/92- ival2 should be integer*2 declaration

* calling argument declarations:
	INTEGER*2 ival2
	INTEGER author_cpu

* internal variable declarations:
        LOGICAL swap

#ifdef unix
      include 'tmap_format/cpu.parm'
#  ifdef sun
      BYTE bb(2),bdum
#  else
      INTEGER*1 bb(2),bdum
#  endif
#else
      BYTE bb(2),bdum
#endif

        INTEGER*2 dum
	EQUIVALENCE (bb(1),dum)

***********   CONVERT between machine types   **************
*** use the pre-processor to select the target word type ***
#ifdef sun
* SUN is the platform in use
      swap = author_cpu .EQ. cptype_vax
     .  .OR. author_cpu .EQ. cptype_dec

#else
#	if unix
* DECstation is the platform in use
      		swap = author_cpu .EQ. cptype_sun

#	else
* VAX is the platform in use
      		swap = author_cpu .EQ. cptype_sun
#	endif
#endif

      IF (swap ) THEN
         dum = ival2
         bdum = bb(1)
         bb(1) = bb(2)
         bb(2) = bdum
         ival2 = dum
      ENDIF

      RETURN
      END
