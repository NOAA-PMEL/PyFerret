	SUBROUTINE ERASE_DSET ( dset )

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
* delete all memory variables with data from a particular data set
* eliminate all context references to it, too

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86
* V200:   6/9/89 - deleted variables flagged in mr_protected
*	10/25/89 - moved memory-resident purging to PURGE_DSET
* V411:  7/95 - Delete variables created with  LET/D=this_dset

#ifdef unix
	include	'ferret.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	dset

* local variable declarations:
	INTEGER mr, ii

* eliminate memory resident data
	CALL PURGE_DSET( dset )

* delete LET-defined variables that belong to this data set
	DO 100 ii = 1, max_uvar
	   IF (uvar_dset(ii) .EQ. dset) uvar_num_items(ii) =
     .					uvar_deleted
 100	CONTINUE


* eliminate context information depending on this data set
	DO 200 ii = 1, max_context
	   IF ( cx_data_set( ii ) .EQ. dset ) THEN
	      cx_data_set( ii ) = unspecified_int4
	      cx_grid( ii ) = unspecified_int4
	   ENDIF
 200	CONTINUE

	RETURN
	END

