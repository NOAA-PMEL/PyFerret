	SUBROUTINE ASSIGN_VARIABLE
     .		      ( cx,
     .			start_block,
     .			num_blocks,
     .			mr,
     .			status      )

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
* expression indicated in the context cx
* the assigned variable will always be designated as "in-progress"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.1 - 1/30/87 - modified for mr_protected declared as INTEGER
* revision 0.2 - 3/23/87 - eliminated mr_norm
* revision 0.3 - 4/1/87  - bad_flag changed
* revision 0.4 - 7/21/87 - contexts changed to reflect exact size requested by
*			   the operator so the stored in-plane memory variable
*			   limits must be rounded out to the grid box edges
* revision 0.5 - 8/18/87 - added trans_arg (for @SHF)
* revision 0.6 - 7/21/88 - find mem var slot with FIND_MR_SLOT
* revision 0.7 - 1/4/89  - added regrid transforms
* V200:  5/18/89 - 4D symmetrical; using mr_del_ deletion chain
*		 - context assignment in a separate routine

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		cx, start_block, num_blocks, mr, status

* internal variable declarations:
	CHARACTER*4	VAR_CODE

* get a free slot in the memory variable table
	CALL FIND_MR_SLOT( cx, mr, status )
	IF ( status .NE. ferr_ok ) RETURN

* point mr slot to data in bulk memory
	mr_blk1     ( mr )	= start_block
	mr_nblks    ( mr )	= num_blocks

* flag it as "in progress"
	mr_protected( mr )	= mr_in_progress

* assign variable context to slot
	CALL ASSIGN_MR_CX( cx, mr )

* success
 5000	RETURN

* error exit : no space available
! 5010	CALL ERRMSG( ferr_too_many_vars, status,
!     .			VAR_CODE(cx_category(cx),cx_variable(cx)), *5000 )

	END
