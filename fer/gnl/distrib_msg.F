	SUBROUTINE DISTRIB_MSG ( )

* display messages pertinent to distribution

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  7/12/91 for Unix version
* V230: 5/4/92 - modified text for politeness
* V312: 5/19/94 - call PROCLAIM from here to simplify the MAIN program

#ifdef unix
	include 'ferret_cmn/xprog_state.cmn'
        include 'ferret_cmn/xrevision.cmn'
#else
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
        INCLUDE 'FERRET_CMN:XREVISION.CMN'
#endif

* calling argument declarations:

* internal variable declarations:

* this message on "BETA" versions
      IF ( progname_mod(1:1) .NE. ' ' ) THEN

	WRITE ( ttout_lun, 3000 ) 
     .    'For information on obtaining a copy of this program contact'
	WRITE ( ttout_lun, 3000 ) 
     .       '   Steve Hankin / ferret@noaapmel.gov / (206)526-6080'
	WRITE ( ttout_lun, 3000 ) 
     .    'Please notify us if you distribute this program'
	WRITE ( ttout_lun, * ) 
      ENDIF

* this message on all versions (5/94)
      CALL PROCLAIM ( ttout_lun, '          ' )


 3000	FORMAT ( ' ***  ',A )

	RETURN
	END
