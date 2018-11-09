    	SUBROUTINE SHOW_DYN_GRIDS(mods_cx)

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
* summarize the dynamic grids and axes currently in existence
* also do sanity checks that use counts are appropriate

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420:  11/95
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V510 *sh* 4/00 - grids may also be allocated by data set variables
*                  Use /DYNAMIC/ALL to see them.
* V532 5/01 *sh* - bug fix: ignore deleted static grids when chking axis usage
* V650  Write to show_lun which may be set to a file unit number rather than std out
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V697  12/15 *acm* Changes to add details on SHOW GRID uvar to indicate 
*                   compressed or subsetted axes
* V710 *acm* 10/16 Ticket 2422: variables uvar_grid, uvar_data_type, uvar_aux_cat, 
*                  uvar_aux_var to be replaced with calls to LIST routines in NCF_Util.c 

* FERRET DOCUMENATION:  dynamic grids and axes
* (entered 1/2016 *sh*)
* There are three types of grids in Ferret:  legacy, dynamic, and temporary
* 1. Legacy grids occupy grid storage belowup to index max_grids
*    They are brute-force allocated using TM_GET_GRIDNUM.
* 2. Dynamic grids occupy grid storage above max_grids.
*    They are managed using routines with "_dyn_" in their names.
*    They appear as a simple table, like the legacy grids, but they are
*    managed through a doubly-linked list into lists of "free" and "used" grids
*    A use count (grid_use_cnt) keeps track of when they are no longer needed.
*    When a use count falls to zer (through de-allocation) a grid is
*    automatically returned to the free list.
*    Each variable in a dataset that depends upon a grid constitutes a "use"
*    as does each memory-resident user variable instance defined on that grid.
*    This routine checks that these counts have not been corrupted.
* 3. Temporary grids are a linked lists of grids that have been temporarily
*    removed from the free and used lists.  They are managed by routines that
*    have "_tmp_" in their names.
*    Grids are put onto the temporary list during the process of initializing
*    a dataset, so that if an error condition occurs during initialization
*    it is simple to clean up the partially created grids.  At successful
*    completion of initializing a dataset all temporary grid should be moved
*    back to the free or used lists. Some go back to the free list because
*    duplicate grid definitions occur temporarily during dataset initialization
* 
* Dynamic management of axes ("lines") is very similar.  Each grid that uses a
*    line bumps the use count for that line.  

        IMPLICIT NONE
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	include 'xtm_grid.cmn_text'
	include	'ferret.parm'
	include	'slash.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
        include 'xfr_grid.cmn'   ! w/ mgrid_abstract
	include 'errmsg.parm'

* calling argument declarations:
	INTEGER mods_cx

* local variable declarations:
        LOGICAL show_all, shown, uvgrid
	INTEGER grid, line, dset, var, count, i, slash_all, status
	INTEGER ngrids, saved_uvgrid
	CHARACTER sanity_msg*24

* local parameter declarations
        PARAMETER ( slash_all  = 1 + 0 )

* initialize
        show_all   = qual_given( slash_all ) .GT. 0
	uvgrid = .FALSE.

* first show the dynamic grids
        shown = .FALSE.
	grid = grid_flink(max_grids)
 	IF ( grid .EQ. max_grids ) THEN
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .						'No dynamic grids',16)
	ELSE
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .						'Dynamic grids:',14)
	ENDIF
* ... --- "while" loop
 100	IF ( grid .NE. max_grids ) THEN
* ... is this an exlicit (netCDF) grid that shouldn't be shown 
           IF (grid_name(grid)(1:1).NE.'(' .AND. .NOT.show_all) THEN
               grid = grid_flink(grid)
               GOTO 100
           ENDIF
* ... sanity check the use count -- must be in use by user vars or dset vars

	   count = 0
           IF (grid .EQ. mgrid_abstract) count = 1  ! is it SET GRID?
	   
	   DO 125 var = 1, max_uvar

              CALL GET_SAVED_UVAR_GRID_LIST_LEN(var, ngrids)
	      IF (ngrids .GT. 0) THEN

		 dset = uvar_dset(var)             ! the dset that owns this LET/D
		 IF (dset .LE. pdset_irrelevant) dset = pdset_uvars  ! global uvars
		 DO i = 1, ngrids
		    CALL NCF_NEXT_UVAR_GRID_IN_LIST ( dset, var, i, saved_uvgrid )
		    IF ( saved_uvgrid .EQ. grid ) count = count + 1
		 ENDDO

	      ENDIF
		    
 125	   CONTINUE

	   DO 135 var = 1, maxvars
	      IF ( ds_grid_number(var) .EQ. grid ) count = count + 1
 135	   CONTINUE
	   IF ( count .EQ. grid_use_cnt(grid) ) THEN
	      sanity_msg = ' '
	   ELSEIF ( count .GT. grid_use_cnt(grid) ) THEN
	      sanity_msg = 'err: grid under-allocated'
	   ELSE
	      sanity_msg = 'err: grid over-allocated'
	   ENDIF
* ... end of sanity check
* 4/99 *kob* modify write and format statement - mostly for stylistic reasons
	   WRITE (risc_buff,3000) 'GRID', 
     .           grid_name(grid),
     .            grid_use_cnt(grid),
     .					sanity_msg
* *kob* 4/99 shorten dynamic grid name on output statment
 3000	   FORMAT ( 4X,A,1X,A16,16X, 'use count:',I4,5x,A)
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
	   CALL SHOW_GRID(grid, mods_cx, uvgrid, status)
	   IF (status .NE. ferr_ok) GOTO 5000
           shown = .TRUE.
	   grid = grid_flink(grid)
	ENDIF
	IF (grid .NE.max_grids) GOTO 100
        IF (.NOT.shown)  CALL SPLIT_LIST(pttmode_explct, show_lun,
     .					'    no implicit grids',21)

* now the dynamic axes
        shown = .FALSE.
	line = line_flink(max_lines)
 	IF ( line .EQ. max_lines ) THEN
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .						'No dynamic axes',15)
	ELSE
	   CALL SPLIT_LIST(pttmode_explct, show_lun,
     .						'Dynamic axes:',13)
	ENDIF
* ... --- "while" loop
 200	IF ( line .NE. max_lines ) THEN
* ... is this an exlicit (netCDF) grid that shouldn't be shown 
           IF (line_name(line)(1:1).NE.'(' .AND. .NOT.show_all) THEN
               line = line_flink(line)
               GOTO 200
           ENDIF
* ... sanity check the use count -- must be in use by user vars
	   count = 0
* ... in use by static grids
	   DO 225 grid = 1, max_grids-1
              IF ( grid_name(grid) .NE. char_init16 ) THEN
	        DO 220 i = 1, nferdims
	           IF ( grid_line(i,grid) .EQ. line ) count = count + 1
 220	        CONTINUE
              ENDIF
 225	   CONTINUE
* ... in use by dynamic grids
	   grid = grid_flink(max_grids)
 230	   IF ( grid .NE. max_grids ) THEN
	      DO 232 i = 1, nferdims
	         IF ( grid_line(i,grid) .EQ. line ) count = count + 1
 232	      CONTINUE
	      grid = grid_flink(grid)
	   ENDIF
	   IF (grid .NE.max_grids) GOTO 230
* ... in use by other axes
	   DO 235 i = max_lines, max_dyn_lines
	         IF ( line_parent(i) .EQ. line ) count = count + 1
 235	   CONTINUE
	   IF ( count .EQ. line_use_cnt(line) ) THEN
	      sanity_msg = ' '
	   ELSEIF ( count .GT. line_use_cnt(line) ) THEN
	      sanity_msg = 'err: axis under-allocated'
	   ELSE
	      sanity_msg = 'err: axis over-allocated'
	   ENDIF
* ... end of sanity check
* *kob* 4/99 - reference format statement at 3000 - for stylistic reasons
	   WRITE (risc_buff,3000) 'AXIS',line_name(line), line_use_cnt(line),
     .					sanity_msg
	   CALL SPLIT_LIST(pttmode_explct, show_lun, risc_buff, 0)
           shown = .TRUE.
	   line = line_flink(line)
	ENDIF
	IF (line .NE.max_lines) GOTO 200
        IF (.NOT.shown)  CALL SPLIT_LIST(pttmode_explct, show_lun,
     .					'    no implicit axes',20)

 5000	RETURN
	END



