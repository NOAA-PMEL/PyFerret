	SUBROUTINE INITIALIZE_FERRET

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
* initialize run-time variables and I/O in preparation to begin execution
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/26/86
* revision 0.1 - 12/11/86 - explicit OPEN of SYS_OUTPUT with 255 byte RECL
*			    and 255 byte RECL on journal file OPEN
* revision 0.2 - 1/30/87 - modified for mr_protected declared as INTEGER
*			   and RADIUS made a parameter in FERRET.PARM
* revision 0.3 - 5/1/87  - EXTERNAL *CONTROL and no LIST file opened
* revision 0.4 - 9/14/87 - added VMS calls to turn on line recall
* revision 0.5 - 11/30/87- added call to enable ^C interrupts
* revision 0.6 - 2/18/88 - converted X axes to "regular" format
* revision 0.7 - 5/2/88  - changed GFDL.JNL to GFDL_JOURNAL (logical) and added
*			   ERR= branch to OPEN
* revision 0.8 - 6/29/88  - changed GFDL_JOURNAL to FERRET_JOURNAL
*			    and today_time in hours:minutes, only
*			    determine mode (interactive) here
* revision 0.9 - 7/21/88 - memory initialization done by INIT_MEMORY
* revision 1.0 - 8/12/88 - added FERRET_INIT optioal initialization file
* revision 1.1 -11/22/88 - new TM IO library
* revision 1.2 - 2/2/89  - allow program to continue w/out journal file
* V200:  6/22/89 - define special grids at run time
*	 12/6/89 - initialize mode_GKS and mode_wait
* Unix/RISC port - 1/91 ... several OS dependencies fixed
*                - 9/91 - fixed err handling when journal file cant be opened
*                - 10/91 - hook to enable PLOT+ braindead features
*                - 10/91 -   FER_INIT ==> .ferret
* V230:  5/14/92 - on SGI (with no backing store) set REFRESH mode
* V300:  5/12/93 - split journal file init into another file
*		 - relocated sgi IFDEF to default REFRESH to ON
*       10/25/93 - default MODE REFRESH ON for **all** systems
*	 2/24/94 - small change due to mode_state as a 2D variable
*	 3/14/94 - base initial state of mode refresh on "ifdef X_REFRESH"
* V312:  5/19/94 - significant portions of routine removed to new MAIN program
* V313: 10/27/94 - *sh* AIX port: avoid the routine name "time"
* V420:  9/19/95 - call TM_INIT_DYN_GRIDS for dynamic (implicit) grid creation
* Linux port *kob* 3/96 - had to add calls to use F90 data and time routine
*		     rather than f77 default routines.
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
*                  Use pseudo-dataset of user-defined axes to keep track of attributes.
* V65   2/10 *acm*- Changes for CLOCK symbols that work with gfortran as well as g77.
*                   see special_symbol.F
* V68   1/12 *acm* declarations changes for double-precision ferret.
*       3/12 *acm* cleanup ifdefs and unnecessary include files
* V698  3/16 *acm* Removed Philander-Seigel code including diagnostic 
*                  variables, pre-defined axes and grids, cat_calc_var.
* V721  8/17 *kms* Restore call to string_array_init with pyvar_code_head
*                  that got lost in the combined rename and update in Ferret

        IMPLICIT NONE
        include 'tmap_errors.parm'  ! error codes
	include 'tmap_dims.parm'
	include 'xio.cmn_text'
        include 'xdset_info.cmn_text'
	external xio_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'gfdl_vms.parm'
	include	'errmsg.parm'
	include	'gfdl.parm'			! from phil/sieg
	include	'xonedim.cmn'		! from phil.sieg
	include 'xprog_state.cmn'
        include 'xalgebra.cmn'
	include 'xvariables.cmn'
	include 'xtoday.cmn'
        include 'xpyvar_info.cmn'

* local variable declarations:
	LOGICAL	GKS_WINDOWS, fexist
	INTEGER	TM_LENSTR1, ito, ifrom, mem_blk, mr, iset, status
      integer icount, irate, imax
	REAL	DTIME, radian, cm2deg
	REAL*4  TArray(2), dtime_res
        CHARACTER*2048 errmsg
        INTEGER errmsglen


*************** I/O ************************************************************
	interactive = .TRUE.

* determine default GKS usage based on system logon device
	mode_GKS = GKS_WINDOWS()
	mode_wait = .NOT.mode_GKS	! no wait after plot on window device

	mode_state( pmode_xwindows, 1 ) = .FALSE.

        CONTINUE
	
*********************** Climatological axes ************************************
	CALL TM_SET_CLIM_AXES

* set up special grids
	CALL DEFINE_SPECIAL_GRIDS   ! label 1000 moved here 9/91 *sh*

* initialize dynamic grid management
	CALL TM_INIT_DYN_GRIDS

*************** TODAY'S DATE ***************************************************

*	CALL DATE( today_date )
*	CALL TIME( today_time )
* *kob*  try new routine for y2k fix
	CALL GET_DATE_AND_TIME (today_date, today_time)
	today_time = today_time(:5)	! w/out seconds

* DTIME returns a real number which is the running time for this 
* process since the last call to DTIME. TIMEARRAY(1) reports the user time, 
* and TIMEARRAY(2) supports the system time. Argument 2 is the sum of the two.
* Call the routine to initialize the result. It can be updated by evaluating 
* the symbol DELTA_CPU. See special_symbol.F

      dtime_res = DTime(TArray)
      cpu_last = 0.0  ! in common xtoday.cmn

* ALso a special symbol CLOCK_SECS, the delta seconds in clock time since Ferret
* session is initialized.  Save clock_start_count to subtract from the value
* when this symbol is updated.

      CALL System_Clock(icount, irate, imax)
      clock_start_count = icount

* initialize pseudo-dataset in linked-list structure which will contain all 
* user variables and their attributes.
* Also another pseudo-dataset to contain user-defined coordinate variables (axes). 

        iset  = -1;
        CALL ncf_init_uvar_dset( iset )
        iset  = -2;
        CALL ncf_init_uax_dset( iset )

*************** STATE VARIABLES ************************************************
* see BLOCK DATA initialization of COMMON/ XPROG_STATE /

*********************** GRIDS **************************************************
c	CALL GFDL_TERMS

	RADIAN=57.29578							! 1027
	cm2deg = radian / radius



*************** MISCELLANEOUS **************************************************
* TMAP librabry routine errors to same output unit as other errors
	lunit_errors = err_lun

*************** INITIALIZE SPECIAL CHARACTERS *************************
*
*       Initialize special characters in COMMON - needed for g77 port to
*         comply with other compilers *kob*  3/03
	pCR = CHAR(15)

*************** BRAINDEAD PLOT+ ******************************************
        CALL PPL_BDEAD

        CALL string_array_init(alg_pvar_head,
     .                         num_pseudo_vars,
     .                         8,
     .                         alg_pvar(1))

        CALL string_array_init(countervar_name_head,
     .                         num_counter_vars,
     .                         128,
     .                         countervar_name(1))

        CALL string_array_init(uvar_name_code_head,
     .                         max_uvar,
     .                         128,
     .                         uvar_name_code(1))

        CALL string_array_init(ds_var_code_head,
     .                         maxvars,
     .                         128,
     .                         ds_var_code(1))

        CALL string_array_init(pyvar_code_head,
     .                         maxpyvars,
     .                         128,
     .                         pyvar_code(1))

        CALL deleted_list_init(uvar_num_items_head,
     .                         uvar_num_items(1),
     .                         max_uvar,
     .                         uvar_deleted)

        CALL FGD_READ_ALL_SYMBOLDEFS(status)
        IF ( status .NE. FERR_OK ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL WARN(errmsg(1:errmsglen))
        ENDIF

	RETURN

	END

