	SUBROUTINE DISP_DATA_SET_UP
     .	( memory, ax1_start, ax1_blks, ax2_start, ax2_blks, status )

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
* set up graphics environment and obtain the data

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 1/7/87 
* revision 0.1 - 2/25/87 - changes for INTERPRETATION STACK
* revision 0.2 - 6/24/87 - changes to accomodate transposed plots
* revision 0.3 - 9/16/87 - added want_cmnd_data to CALL GET_CMND_DATA
* V200:  5/19/89 - 4D symmetrical
*         5/9/90 - allow CGRID_SIZE for axes to accomodate PLOT/VS
* V301:   1/7/94 - don't unprotect the command data cuz' GET_MEMORY is called
*		   during set-up of plots
* V312: 5/94 - array "memory" as a calling argument
* V500: 2/99 *sh* - added support for POLYGON command -- working storage needs

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/slash.parm'
	include 'ferret_cmn/command.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:SLASH.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif
	
* calling argument declarations:
	INTEGER	ax1_blks, ax1_start, ax2_blks, ax2_start, status
	REAL memory(*)

* internal variable declarations:
	INTEGER	MR_DIM_LEN, CGRID_SIZE,
     .		idim, ndim, dim(4), arg_num, mr1, ax_len, cx

* get the requested data somehow  (and leave it protected 1/94)
	CALL GET_PROT_CMND_DATA ( memory, cx_last, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* compute working storage
	IF (cmnd_num .EQ. cmnd_polygon) THEN
* ... allocate buffers used to extract individual polygon coord arrays
*     set buffer size to the longest axis of 1st coord arg
	   cx = is_cx(1)
	   CALL GET_CX_DIMS(cx, ndim, dim)
	   ax_len = 1
	   DO 100 idim = 1, ndim
 100	   ax_len = MAX(ax_len,
     .			cx_hi_ss(cx,dim(idim)) - 
     .			cx_lo_ss(cx,dim(idim)) + 1 )
	ELSE
* ... allocate memory used for single precision axes
*     make them each equal to MAX in case /TRANSPOSE or mulit-dim PLOT/VS
	   mr1 = is_mr( 1 )
	   CALL GET_CX_DIMS( is_cx(1), ndim, dim )
	   ax_len = MAX( MR_DIM_LEN( dim(1), mr1 ),
     .		         MR_DIM_LEN( dim(2), mr1 ),
     .		         CGRID_SIZE( is_cx(1) )    )
	ENDIF

* allocate working storage
	CALL GET_MEMORY( ax_len, ax1_start, ax1_blks, status )
	IF ( status .NE. ferr_ok ) GOTO 5020
	CALL GET_MEMORY( ax_len, ax2_start, ax2_blks, status )
	IF ( status .NE. ferr_ok ) GOTO 5010

* successful completion
	RETURN

* error exits - release the memory used for axis buffers
! 5000	CALL FREE_MEMORY( ax2_start, ax2_blks )
 5010	CALL FREE_MEMORY( ax1_start, ax2_blks )
 5020	CALL UNPROTECT_CMND_DATA	! added 2/99
 5100	RETURN
	END
