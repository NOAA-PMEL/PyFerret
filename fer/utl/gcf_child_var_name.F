	SUBROUTINE GCF_CHILD_VAR_NAME( parent_uvar, char1, name )

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
* return the (hidden) name of a child variable constructed from
* the number of its parent variable and the start character of the
* grid-changing reference within the parent definition

* For example, in the definition
*     yes? LET var = 2 * GC_FUNCT( a,b,c )
* the string "GC_FUNCT( a,b,c )" would become a child variable with a name
* based on the uvar number of "var" and a start char of 5 (following "2 * ")

* V450: 12/96
*        7/7/97 - revert to one name in all cases -- use DIAGNOSTIC_OUT
*                 to make diagnostic output more readable

* calling argument declarations
	INTEGER		parent_uvar, char1
	CHARACTER	name*(*)

* internal variable declarations

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* sanity check
	IF ( LEN(name) .LT. 11 ) THEN
	   STOP 'GCF_CHILD_VAR_NAME'
	ENDIF

!	IF ( char1.LT.100 .AND. parent_uvar.LT.100) THEN
!	   WRITE (name, 3000) char1, parent_uvar
! 3000	   FORMAT ('(C',I2.2,',V',I2.2,')')	! e.g. "(C12,V02)"
!	ELSE
	   WRITE (name, 3100) char1, parent_uvar
 3100	   FORMAT ('(C',I3.3,',V',I3.3,')')	! e.g. "(C112,V002)"
!	ENDIF

	RETURN
	END
