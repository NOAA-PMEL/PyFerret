	SUBROUTINE SHOW_MEM_VARS

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
* execute the SHOW VARIABLES/MEMORY command - display a priority-sorted list 
* of the data in memory

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 5/29/87 - (from XEQ_EXAMINE)
* revision 0.1 - 7/8/87  - made mr_list and mr_sort into dummy args
* revision 0.2 - 9/17/87 - made permanent vars,only the default
*			   and added trans_arg output
* revision 0.3 - 3/25/88 - added /NAMED qualifier, reformatted "PERM" output
*			   and added interrupts
* revision 0.4 - 6/27/88 - added /PERM and /TEMP qualifiers
* revision 0.5 - 10/5/88 - loop 2,max_mrs --> 1,max_mr_avail
* revision 0.6 - 1/10/89 - added grid info
* V200:  5/19/89 - 4D symmetrical; major reformatting of output
* 		 - deleted variables flagged in mr_protected
*		 - sorting not needed because of deletion priority chain
*		 - most code moved to SHOW_1_MR
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
* V650  Write to show_lun which may be set to a file unit number rather than std out*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xinterrupt.cmn'

* calling argument declarations:

* local variable declarations:
	LOGICAL	show_all, show_perm, show_named, show_temp
	INTEGER status, iqual, mr
	CHARACTER datmsg*16
	PARAMETER ( datmsg = 'data loaded with' )	!kob 12/96
* local parameter declarations:
	INTEGER	    slash_all, slash_named, slash_temp, slash_perm
	PARAMETER ( slash_all = 1,
     .		    slash_named = 2,
     .		    slash_temp  = 3,
     .		    slash_perm  = 4 )

* interpret qualifiers
	show_all   = qual_given( slash_all ) .GT. 0
	show_named = show_all .OR. qual_given( slash_named ) .GT. 0
	show_perm  = show_all .OR. qual_given( slash_perm ) .GT. 0
	show_temp  = show_all .OR. qual_given( slash_temp ) .GT. 0
	show_all   = show_temp .AND. show_perm
* ... /PERM is default
	show_perm  = show_perm .OR.
     .		     .NOT.(show_all .OR. show_named .OR. show_temp)

* write introductory line
	IF ( show_all ) THEN
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .				'all data in memory:',19 )
	ELSE
	   IF ( show_named ) CALL SPLIT_LIST(pttmode_explct, show_lun,
     .				datmsg//' /NAME=:',24 )
	   IF ( show_perm  ) CALL SPLIT_LIST(pttmode_explct, show_lun,
     .				datmsg//' /PERMANENT:',28 )
	   IF ( show_temp  ) CALL SPLIT_LIST(pttmode_explct, show_lun,
     .				datmsg//'out /PERMANENT:',31 )
	ENDIF

* display permanent variables
	IF ( show_perm ) THEN
	   DO mr = 1, max_mr_avail
	      IF ( interrupted ) CALL ERRMSG( ferr_interrupt,status,' ',*5000 )
	      IF (mr_protected(mr).EQ.mr_perm_protected) CALL SHOW_1_MR(mr,status)
 	   ENDDO
	ENDIF

* display deletable variables
	IF ( show_temp ) THEN
	   mr = mr_del_flink(0)
 200	   IF ( mr .NE. 0 ) THEN
	      IF ( interrupted ) CALL ERRMSG( ferr_interrupt,status,' ',*5000 )
	      CALL SHOW_1_MR( mr, status )
	      IF (status .NE. ferr_ok) GOTO 5000
	      mr = mr_del_flink(mr)
	      GOTO 200
	   ENDIF
	ENDIF

 5000	RETURN
	END
