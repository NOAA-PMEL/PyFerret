	SUBROUTINE RE_ASSIGN_VARIABLE( mr, new_cx )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* re-assign the data which is currently cataloged in the memory variable
* table under mr to the context information stored in new_cx

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/19/87
* V200:  6/20/89 - improved deletion priority scheme
*		 - variable is assumed NOT to be in deletion chain
*		 - use ASSIGN_MR_CX
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V702:  3/17 *sh* dynamic memory management

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'

* calling argument declarations:
	INTEGER mr, new_cx

* internal variable declarations:
	INTEGER HASH_CX, prev, next, pile, head

* this should never happen, but to be safe ...
	IF (mr_protected(mr) .EQ. mr_not_protected) THEN
	   CALL WARN("crptn??: re_assign_variable")
	   essential_mem = essential_mem + mr_size(mr)
	   IF (essential_mem.GT.peak_essential_mem) 
     .		peak_essential_mem = essential_mem
	ENDIF

* flag the variabe as in progress
	mr_protected(mr) = mr_in_progress

* make this mem var the property of the new context
	CALL ASSIGN_MR_CX( new_cx, mr )

* correct the links to put this in the right hash pile
*    ... remove variable slot from hash pile doubly linked list
	   prev = mv_blink( mr )
	   next = mv_flink( mr )
	   mv_flink( prev ) = next
	   mv_blink( next ) = prev
*    ... add the slot to the head of the correct hash pile
	   pile = HASH_CX( new_cx )
	   head = mv_flink( pile )
	   mv_flink( pile ) = mr
	   mv_flink( mr ) = head
	   mv_blink( mr ) = pile
	   mv_blink( head ) = mr

	RETURN
	END
