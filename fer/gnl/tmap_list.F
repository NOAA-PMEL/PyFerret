	SUBROUTINE TMAP_LIST ( memory, lun, reclen, nvars, mr_list,
     .			       cx_list, rec_buf, status )

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
* create a complete time step of 1 to several variables (all given in a single
* command) in GT format including the STEP record

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/7/88
* revision 1.0 - 6/24/88 - major mods. to permit multiple variables
* V200:  5/24/89 - 4D symmetrical
*	10/16/89 - set up adjustable array limits with NON_ARRAY_SUBSC
*	12/19/89 - eliminated the possibility of multiple data set variables
* RISC/Unix port *sh* 5/91:  include machine type indicator in output STEP
*                - output using direct access (note: error if appended)
*           8/91 - machine cpu type was wrong - fixed it using "current_cpu" 
* SUN port 3/92 *sh* a couple of ifdef's for record length and logical*1
* V312: 5/94 - array "memory" as a calling argument
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
*  8/05 *acm* - Allow more digits in xrevision number

	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'
	include 'xrevision.cmn'
	include 'xinterrupt.cmn'
        external xmachine_data

#ifdef sun 
        include 'xmachine_byte.cmn'
        BYTE i1zero/0/
#else
        include 'xmachine_int1.cmn'
        INTEGER*1 i1zero
	PARAMETER (i1zero = 0)
#endif

* calling argument declarations:
	INTEGER		lun, reclen, nvars,
     .			mr_list(nvars), cx_list(nvars), status
	REAL	memory( mem_blk_size, max_mem_blks )

* internal variable declarations:
	INTEGER*2 	nvari
	INTEGER		CX_DIM_LEN, CGRID_AXIS, 
     .			nchar64, cx, mr, cx1, mr1, dset1, grid,
     .			nrecperstep, i, taxis, ndim, dim(4), l, recnum,
     .                  slen
	REAL		rec_buf(*)
	REAL*8		TM_WORLD
	CHARACTER	COMPOUND_KEY*255, TM_FMT*12,
     .                  char64*128, mod_num*12, run_num*4

* initialization
	cx1	= cx_list(1)
	mr1     = mr_list(1)
	grid	= cx_grid( cx1 )
	nvari	= nvars
	taxis	= CGRID_AXIS( t_dim, cx1 )

* determine 4D shape and joint context for all variables to be output
	CALL GRP_CX_DOC( cx_list, nvars, cx_buff, cx_xtra_doc )
	CALL GET_CX_DIMS( cx_list(1), ndim, dim )
	IF ( ndim.EQ.1 .AND. dim(1).EQ.t_dim ) GOTO 5010

* all variables must be at same times
	DO 100 i = 2, nvars
	   cx = cx_list(i)
	   IF ( CGRID_AXIS( t_dim, cx ) .NE. taxis
     .	   .OR. cx_lo_s4( cx ) .NE. cx_lo_s4( cx1 )
     .	   .OR. cx_hi_s4( cx ) .NE. cx_hi_s4( cx1 ) ) GOTO 5020
 100	CONTINUE

* experiment number, if known
	dset1   = cx_data_set( cx_buff )
	run_num = '    '
	IF ( dset1 .EQ. pdset_irrelevant ) THEN
	   run_num = '    '
	ELSEIF ( dset1 .EQ. unspecified_int4 ) THEN
	   run_num = 'mult'
	ELSE
	   run_num = ds_expnum( dset1 )
	ENDIF

* use program revision level as mod number
 210	mod_num = TM_FMT(revision_level, 5, 12, slen)

* comments for STEP record (all axes documented)
	DO 220 i = 0, 10
 220	cx_xtra_doc(i) = .TRUE.
	char64 = COMPOUND_KEY( cx_list, nvars, cx_xtra_doc, nchar64 )

* nchar can be the largest multiple of 4 that will fit
* modified below ifdef logic.  reclen_in_bytes is the correct ifdef test
* not just "sun"  *kob* 5/97
#ifdef sun
# 	define reclen_in_bytes
#endif
#ifdef reclen_in_bytes
	nchar64 = MIN( 128, reclen-28, nchar64 )  ! record length in bytes on sun
#else
	nchar64 = MIN( 128, 4*reclen-28, nchar64 )
#endif
	nchar64 = 4*(nchar64/4)

* number of records per step
	nrecperstep = 1
	DO 400 i = 1, nvars
	   cx = cx_list(i)
	   nrecperstep = nrecperstep
     .		    + CX_DIM_LEN(y_dim,cx) * CX_DIM_LEN(z_dim,cx)
 400	CONTINUE

* LOOP THROUGH THE TIME STEPS
        recnum = 1
	DO 600 l = cx_lo_s4(cx1), cx_hi_s4(cx1)
	   IF ( interrupted ) RETURN	! always stop at complete time steps

* write 'STEP' record
	   WRITE ( UNIT=lun, REC=recnum  )
     .			'STEP',
     .			TM_WORLD( l, grid, t_dim, box_middle ),
     .			nrecperstep,
     .			nvari,
     .                  active_cpu,       ! different depending on host cpu
     .                  i1zero,           ! reserved for compression flag
     .			run_num,
     .			mod_num,
     .			char64(:nchar64)
           recnum = recnum + 1

* now the data from all of the variables
	   DO 500 i = 1, nvars
	      mr = mr_list(i)
	      cx = cx_list(i)
	      CALL NON_ARRAY_SUBSC( mr, 1 )
	      CALL GTVAR_LIST( lun,memory(1,mr_blk1(mr)),mr,cx,l, recnum )
 500	   CONTINUE

 600	CONTINUE

	status = ferr_ok
	RETURN

* error exit(s)
 5000	RETURN
 5010	CALL ERRMSG( ferr_dim_underspec, status,
     .		     'use TS format for time series', *5000 )
 5020	CALL ERRMSG( ferr_dim_underspec, status,
     .				 'times differ between variables', *5000 )
	END
