	SUBROUTINE DISP_DATA_SET_UP
     .	( memory, ax1_start, ax1_blks, ax2_start, ax2_blks, status )

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

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/slash.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
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
	INTEGER	MR_DIM_LEN, CGRID_SIZE, ndim, dim(4), arg_num, mr1, ax_len

* get the requested data somehow  (and leave it protected 1/94)
	CALL GET_PROT_CMND_DATA ( memory, cx_last, status )
	IF ( status .NE. ferr_ok ) GOTO 5100

* allocate memory used for single precision axes
* make them each equal to MAX in case axes get transposed or mulit-dim PLOT/VS
	mr1 = is_mr( 1 )
	CALL GET_CX_DIMS( is_cx(1), ndim, dim )
	ax_len = MAX( MR_DIM_LEN( dim(1), mr1 ),
     .		      MR_DIM_LEN( dim(2), mr1 ),
     .		      CGRID_SIZE( is_cx(1) )    )
	CALL GET_MEMORY( ax_len, ax1_start, ax1_blks, status )
	IF ( status .NE. ferr_ok ) GOTO 5100
	CALL GET_MEMORY( ax_len, ax2_start, ax2_blks, status )
	IF ( status .NE. ferr_ok ) GOTO 5010

* successful completion
	RETURN

* error exits - release the memory used for axis buffers
! 5000	CALL FREE_MEMORY( ax2_start, ax2_blks )
 5010	CALL FREE_MEMORY( ax1_start, ax2_blks )
 5100	RETURN
	END
