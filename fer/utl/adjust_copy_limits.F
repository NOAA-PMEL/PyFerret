	SUBROUTINE ADJUST_COPY_LIMITS (axis, mchunk, sorted_indices,
     .				       phase, max_phase)

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
* a "chunk" of source data is available. The variable sorted_indices contains
* a list of subscript values to be pulled from that source data . The current
* value of "phase" indicates the index of the highest sorted_index that
* we intend to use from the chunk of source data. In this routine we
* check to see if more source data might be available that was expected, and
* if so, we adjust the phase upward to utilize all of what's available
* ==> this routine is solely to optimize performance

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V510: 10/99 *sh*

	include 'ferret.parm'
!	include	'xcontext.cmn'
	include	'xvariables.cmn'

* calling argument declarations:
	INTEGER  axis, mchunk, phase, max_phase
	REAL    sorted_indices(max_phase)


* internal variable declarations
	INTEGER max_available, lo_phase, hi_phase, test

* A typical scenario for the use of this routine might be as follows:
*   sorted list of subscripts at which to sample: {11,12,14,16,18,20}
*   souce data along axis extends from 1:19
*   current phase=3 (sorted_indices(phase)=14)
*   so we should adjust phase to a new value of 5 (sorted_indices(phase)=18)

* initialize
	max_available = mr_hi_ss(mchunk,axis)

* screen out obvious cases
	IF (sorted_indices(phase) .GE. max_available) THEN
* ... current phase is already as high as it can be
	   RETURN   ! phase is OK as-is
	ELSEIF (sorted_indices(max_phase) .LE. max_available) THEN
* ... can utilize the entire chunk of data available
	   phase = max_phase
	   RETURN
	ENDIF 

* initialize search for max that can be used from the chunk
	lo_phase = phase
	hi_phase = max_phase

* binary search 
 100	test = (lo_phase+hi_phase)/2
	IF (sorted_indices(test) .GT. max_available) THEN
	   hi_phase = test
	   GOTO 100
	ELSEIF (sorted_indices(test+1) .GT. max_available) THEN
	   phase = test
	   RETURN
	ELSE
	   lo_phase = test
	   GOTO 100
	ENDIF

	END 
