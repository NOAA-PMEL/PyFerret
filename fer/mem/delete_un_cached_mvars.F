	SUBROUTINE DELETE_UN_CACHED_MVARS

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
* clear (delete) the entire list of uncached memory variables
* (sanity check along the way)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420: 11/95
* V510: 8/99 *sh* -- added deletion of pmv_work_var_pile

#ifdef unix
	include 'ferret.parm'
	include 'xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
* ...

* internal variable declarations:
	INTEGER	mr

* initialize linked list search of "nocache" variables
	mr =  mv_flink( pmv_nocache_pile )

* gobble up the linked list
 100	IF ( mr .EQ. pmv_nocache_pile ) THEN
	   RETURN
	ELSE
	   CALL DELETE_VARIABLE( mr )		! delete head of list
	   mr =  mv_flink( pmv_nocache_pile )	! new head of list
	   GOTO 100
	ENDIF
	   
* ---

* initialize linked list search of "work_var" variables
	mr =  mv_flink( pmv_work_var_pile )

* gobble up the linked list
 200	IF ( mr .EQ. pmv_work_var_pile ) THEN
	   RETURN
	ELSE
	   CALL DELETE_VARIABLE( mr )		! delete head of list
	   mr =  mv_flink( pmv_work_var_pile )	! new head of list
	   GOTO 200
	ENDIF
	   
	END





