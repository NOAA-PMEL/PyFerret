	SUBROUTINE XEQ_USER_COMMAND( memory )

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
* execute the USER_COMMAND command
* this is a template routine that can be modified to include any code of
* a user's choice, thereby customizing the FERRET program

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  1/21/90
* V301:  11/93 - bug fix:  was passing element 2 of var 2 element 3 of 3, etc.
*			: command string decoded by user routines
*         1/94 - need to pass at least 5 arguments for COMMAND=SAMPLE
* V312: 5/94 - array "memory" as a calling argument
*	(and passed to USER_SUB - fixed 6/1/94)

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	REAL	memory( mem_blk_size, max_mem_blks )

* local variable declarations:
	LOGICAL	cmnd_given
	INTEGER	mr, cx, ivar, mr_list(6), cx_list(6), loc, status
	
* get the requested data somehow - possibly several grids of data
	CALL GET_CMND_DATA ( memory, cx_last, status )
	IF ( status .NE. ferr_ok ) GOTO 5000

* set up dummies if fewer than 4 components
	DO 100 ivar = 1, num_uvars_in_cmnd
	  mr_list(ivar) = is_mr(ivar)
	  cx_list(ivar) = is_cx(ivar)
 100	CONTINUE
	DO 110 ivar = num_uvars_in_cmnd+1, 6    ! 4->6 2/25 bug found on SUN
	  mr_list(ivar) = dummy_mr
	  cx_list(ivar) = cx_buff
 110	CONTINUE

* compute and display the results
	CALL NON_ARRAY_SUBSC( mr_list, num_uvars_in_cmnd )

* pass control to the user's routine
	CALL USER_SUB(    memory,
     .			  memory(1, mr_blk1(mr_list(1))), 
     .			  memory(1, mr_blk1(mr_list(2))), 
     .			  memory(1, mr_blk1(mr_list(3))), 
     .			  memory(1, mr_blk1(mr_list(4))), 
     .			  memory(1, mr_blk1(mr_list(5))), 
     .			  memory(1, mr_blk1(mr_list(6))), 
     .			  mr_list, cx_list, num_uvars_in_cmnd,
     .			  ttout_lun, status )
	RETURN

* error exit(s)
 5000	RETURN
	END
