	SUBROUTINE CD_SCAN_VARS(dset,cdfid,reversed,tregular,status)

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
* Loads common blocks XDSET_INFO, XSTEP_FILES, and grid COMMON with vital
* system and background information from a netCDF file
* It processes FERRET special attributes without regard to whether variables
* are "generic" or "FERRET"

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92 - adds a new data set format to TMAP libraries
*              - based loosely on EZ_INIT_DSET
*      4/18/92 - replaced "elif" syntax with
*                   else
*                     if
*                for sgi port   *kob*
*      8/18/92 - variable name with attribute error messages
*       3/4/93 - read global bad data flag (for EPIC files)
*              - don't process EPIC variable 'time2'
*	  5/94 - added support for DOUBLE variables
*	 11/94 - ignore the "history" attribute.  Use "long_name_mod" instead.
*	  2/95 - check data type of missing_value.  Possibly convert
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* 7/98 *sh* - allow broader flexibility in variable names through retaining
*             original case when appropriate
* 1/99 *sh* - fixed bug introduced in 7/98 changes -- blows up when file has
*		only coord variables (e.g. climatological_axes)
* 2/99 *kob* - adding call to SWITCH_NAN for NaN recognition
* V500 3/99 *sh* -- most of this routine removed to cd_get_1_dep_var
*	and this routine consolidated with cd_get_axes. The consolidation
*	allows us to process invalid axes as dependent variables rather than
*	rejecting the netCDF file 
* 3/99 *kob* - make sure to pass the correct string length for newly upped
*              parameter values
* V510 *sh* 3/00: added tregular argument
*		- convert to dynamic grid and axis management

* argument definitions:
*       dset    - pointer to TMAP data set
*	cdfid   - id of currently open CDF file
*	reversed    - logical array to record which axes have reversed coords
*	status	- return status

* include files
#ifdef unix
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
	include 'xdset_info.cmn_text'

	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xbuild_grids.cmn'

	external xdset_info_data
        include 'xdset_info.cd_equiv'
        include 'xio.cmn_text'
        external xio_data
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
#include "netcdf.inc"
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CD_EQUIV'
        INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
        EXTERNAL XIO_DATA
#endif

* argument declarations:
	LOGICAL        reversed(max_lines:line_ceiling), tregular
	INTEGER	      dset, cdfid, status

* local parameter definitions:
	LOGICAL	    do_warn
	INTEGER     str_eq
	PARAMETER ( str_eq = 0,
     .		    do_warn = .TRUE. )

* local variable declarations:
*  ivtm - TMAP library variable ID
*  ivcd - netCDF file variable ID
      LOGICAL CD_GET_ATTRIB, CD_GET_ATTVAL, CD_GET_BOUNDS,
     .        TM_LEGAL_NAME,
     .        got_it, got_bad, got_missing, tf2, its_epic, 
     .        got_a_variable, failed_axis
      INTEGER NCVID,
     .	      STR_UPCASE, STR_CASE_BLIND_COMPARE, TM_LENSTR1, NCDID,
     .        ndims, nvars, cdfstat, las_pos, first_pos, idim, iaxis,
     .        ivcd, ivtm, vartyp, nvdim, vdims(8), nvatts, i, vlen,
     .        ngatts, recdim, istat, npts, lasdim, attype, epic_t2
* 4/99 *kob*
      CHARACTER    vname*64, name*16, buff*64
      CHARACTER*13 TM_STRING
      REAL*4  global_bad

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* special equivalence (cannot include in tmap_format/xdset_info.cd_equiv because "BYTE"
* isn't supported on DECstation
* reorder loop logic because linux cpp didn't like it *kob* 10/96

#ifdef unix
#ifdef sun
      BYTE cd_data_type(maxvars)
#else
      INTEGER*1 cd_data_type(maxvars)
#endif
#else
      BYTE cd_data_type(maxvars)
#endif
      EQUIVALENCE ( ds_precision, cd_data_type )
      REAL    r4_grid_start(4,maxvars),
     .        r4_grid_end(4,maxvars)
      EQUIVALENCE ( r4_grid_end,   ds_grid_end ),
     .            ( r4_grid_start, ds_grid_start )

* initialize: save the state of grid memory prior to any new definitions
* ... save original start of free line memory
	orig_line_mem = next_line_mem_pos

* ... flag all axes as non-reversed as the default
	DO 10 i = max_lines, line_ceiling
 10	reversed(i) = .FALSE.

!* ... how many grid spaces used before we start ?
!	status = TM_FIND_GRID_SLOT ( firstgrid )
!        lastgrid = firstgrid - 1
!	IF (status .NE. merr_ok) GOTO 5900
!
!* ... how many line spaces used before we start ?
!	status = TM_FIND_LINE_SLOT ( firstax )
!        lastax = firstax - 1
!	IF (status .NE. merr_ok) GOTO 5900

* get the vital statistics for the CDF file
        CALL NCINQ( cdfid, ndims, nvars, ngatts, recdim, cdfstat )
        IF ( cdfstat .NE. NCNOERR ) GOTO 5200
        its_epic = ds_type(dset) .EQ. 'ECDF'

* get the global bad data flag (used by EPIC but sometimes missing)
        got_bad = CD_GET_ATTVAL( cdfid, NCGLOBAL, 'missing_value',
     .             do_warn, ' ', global_bad, 1, istat )
        IF ( .NOT.got_bad ) THEN
           IF ( its_epic ) THEN
              global_bad = 1.E35     ! default for EPIC
           ELSE
              global_bad = -1.E34    ! default for FERRET
           ENDIF
        ENDIF

*   * * * loop through all variables in the net CDF file * * *
* classify each as a coordinate variable (axis) or as a dependent
* variable. If a coordinate variable then TRY to interpret it,
* (in cd_get_1_axis) but if it proves to be invalid, then process
* it, instead, as a dependent varoable
	las_pos  = 1    ! reduces redundant search for space
	got_a_variable = .FALSE.
	DO 1000 ivcd = 1,nvars

* initialize
	   failed_axis = .FALSE.

* get the vital statistics for this variable
           CALL NCVINQ(cdfid, ivcd, vname, vartyp, nvdim, vdims,
     .                 nvatts,cdfstat)
           IF ( cdfstat .NE. NCNOERR ) GOTO 5200
           vlen = TM_LENSTR1(vname)


*** CHECK A NUMBER OF REASONS TO IGNORE THIS VARIABLE
* is the dimension name "grid_definition" ? If so, handled elsewhere
	   CALL NCDINQ( cdfid, vdims, buff, npts, cdfstat )
           istat = STR_CASE_BLIND_COMPARE('grid_definition', buff(:15) )
           IF ( istat .EQ. str_eq ) GOTO 1000

* or is it milliseconds for an an EPIC time axis? If so, handled with "time"
           IF ( its_epic .AND. vname(:vlen).EQ.'time2' ) GOTO 1000

* Is it an edge definition ? If so, process w/ corresponding axis
           got_it = CD_GET_ATTRIB( cdfid, ivcd, 'edges',
     .                            do_warn, vname(:vlen), buff, 64 ) !linenamlen
           IF ( got_it .AND. buff.EQ.' ' ) GOTO 1000

* or is it really a child axis (avail for. generic CDF apps but not needed) ?
           got_it = CD_GET_ATTRIB( cdfid, ivcd, 'child_axis',
     .                            do_warn, vname(:vlen), name, 16 ) !linenamlen
           IF ( got_it ) GOTO 1000


*** DECIDE IF IT IS AN AXIS OR A DEPENDENT VARIABLE
* if it is not 1D it must be a dependent variable (3/99)
	   IF ( nvdim .NE. 1 ) GOTO 500

* if for some reason it is desirable for an axis name not to match its dim
* name then the "parent_axis" attribute can flag that it is a parent axis 
           got_it = CD_GET_ATTRIB( cdfid, ivcd, 'parent_axis',
     .                            do_warn, vname(:vlen), name, 16 )
           IF ( got_it ) GOTO 100

* if var name doesn't match a dim name it isn't a coordinate var
	   i = NCDID( cdfid, vname(:vlen), cdfstat )
           IF (cdfstat .NE. NCNOERR ) GOTO 500

* PROCESS AN AXIS	   
 100        CALL CD_GET_1_AXIS(cdfid, ivcd, iaxis,
     .                         its_EPIC, reversed, tregular, status)
           IF ( status .NE. merr_ok ) GOTO 5900
	   IF (iaxis .NE. 0) GOTO 1000		! iaxis>0 if axis was made
	   failed_axis = .TRUE.			! go on and treat as a dep. var

* PROCESS A DEPENDENT VARIANLE
 500	   CALL CD_GET_1_DEP_VAR ( dset, cdfid, ivcd, las_pos, global_bad, 
     .				   ivtm, status )
	   IF (status .NE. merr_ok) GOTO 5900
	   IF ( ivtm .NE. 0 ) THEN
	      las_pos = ivtm	! optimizes next hunt for slot
	      IF (.NOT.got_a_variable) THEN
	         first_pos = ivtm
	         got_a_variable = .TRUE.
	      ENDIF
	   ENDIF
	   IF ( failed_axis ) ds_var_titl_mod(ivtm) = 
     .				'invalid coordinate axis'

* EPIC KLUDGE: if an EPIC file has an invalid "TIME" axis we want, not only
* "TIME" as a dependent variable, but also "TIME2"
	   IF ( its_epic .AND. vname(:vlen).EQ.'time' ) THEN
	      epic_t2 = NCVID( cdfid, 'time2', cdfstat )
	      IF ( cdfstat .EQ. NCNOERR ) THEN
	        CALL CD_GET_1_DEP_VAR (dset,cdfid, epic_t2,
     .			las_pos, global_bad, ivtm, status )
	        IF (status .NE. merr_ok) GOTO 5900
	        IF ( ivtm .NE. 0 ) THEN
	          las_pos = ivtm	! optimizes next hunt for slot
	          IF (.NOT.got_a_variable) THEN
	            first_pos = ivtm
	            got_a_variable = .TRUE.
	          ENDIF
	        ENDIF
	         ds_var_titl_mod(ivtm) = 'invalid EPIC time axis'
	     ENDIF
	   ENDIF

 1000    CONTINUE
*   * * * * * * * * * * * * * * * * * * * * * * * * 

* pass through the variable names making sure that case-sensitive
* duplicates (e.g. 'var' and 'VAR') retain case and that names which
* would be illegal variable names to Ferret are retained in original form
	IF (got_a_variable) THEN
	  DO 310 ivtm = first_pos,las_pos
	    IF (ds_var_setnum(ivtm) .NE. dset) GOTO 310
	    IF (.NOT.TM_LEGAL_NAME(ds_var_code(ivtm))) GOTO 310  ! leave as-is
* ... see if this name case-blind duplicates any other in the data set
	    DO 300 i = first_pos,las_pos
	      IF ( i .EQ. ivtm ) GOTO 300
	      istat = STR_CASE_BLIND_COMPARE(ds_var_code(ivtm),
     .                                       ds_var_code(i)    )
	      IF ( istat .EQ. str_eq ) GOTO 310                  ! leave as-is
 300	    CONTINUE
* ... (finally) upcase the name
	  istat = STR_UPCASE( ds_var_code(ivtm), ds_var_code(ivtm) )
 310    CONTINUE
	ENDIF

* successful completion
	status = merr_ok
	RETURN

* ERROR MESSAGES
 5200   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_SCAN_VARS', cdfid,
     .        no_varid, no_errstring, no_errstring, *5900 )

* error exit
 5900   RETURN
	END
