	SUBROUTINE DISTRIB_MSG ( )

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
* display messages pertinent to distribution

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  7/12/91 for Unix version
* V230: 5/4/92 - modified text for politeness
* V312: 5/19/94 - call PROCLAIM from here to simplify the MAIN program

#ifdef unix
	include 'xprog_state.cmn'
        include 'xrevision.cmn'
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
