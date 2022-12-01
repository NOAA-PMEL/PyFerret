	SUBROUTINE CD_SCAN_VARS(dset,cdfid,reversed,tregular,use_strict,status)

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
*  derivative works and supporting documentation.  Further, the user5/4/2005
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
* V552 5/03 *acm*: increase size of vname from 64 to 128 chars
* V600 8/05 *amc*  Fix bug 1318, wrong string length of bname in last arg to CD_GET_ATTRIB
* V600 *ACM* attribute control. 
*             Change call to CD_GET_ATTRIB to NC_GET_ATTRIB - get attrib from 
*               linked list structure
*             CD_GET_ATTVAL replaced by NC_GET_ATTRIB, which now returns  
*               strings or real values according to attrib type
*             Replace NCINQ with NCF_INQ_DS
*             Replace NCVINQ for variables, with CD_GET_VAR_INFO
*             Replace NCDID with CD_GET_DIM_ID
*             Replace NCDINQ with CD_GET_DS_DIMS
* V601 *acm*  Fix bug 1445: if there were no global attributes, then the flag 
*             global_bad was not set (defaulted to 0). so for a netcdf file 
*             without global attributes, variables having no missing value 
*             flags got bad-flag of 0.
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* V67   1/11 *acm* Fix ticket 1777. If a coordinate variale is STRING, issue notes.
* V67   1/11 *acm* Fix ticket 1794. If no missing/fillvalue, issue a warning ONLY
*             when diagnostic mode is set.
* v67   2/11 *acm* Do the notes about no missing/fillvalue for each variable,
*             when in diagnostic mode. So move that back to cd_get_1_dep_var.
* V68   4/11 *acm* fix 1848: skip vars with dim>4 or string vars with dim>5
* V68  *acm* 1/12- added use_strict argument for ticket 1910: micro-adjusting on 
*                  repeated coordinates in netCDF/OPeDNAP datasets
* V6.74 3/12 *acm* cleanup ifdefs and unnecessary include files
* V6.87 4/14 *acm* ticket 2146: if invalid bounds, make bounds a depend. var
* V695+ 4/15 *sh* support for auxiliary vars (coordinates attributes)
* v7.6 *acm* 5/20 Issue 1876: can read string-typed data, write as char.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 


* argument definitions:
*       dset    - pointer to TMAP data set
*	cdfid   - id of currently open CDF file
*	reversed    - logical array to record which axes have reversed coords
*	status	- return status

* include files
        include 'netcdf.inc'
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

* argument declarations:
	LOGICAL        reversed(max_lines:line_ceiling), tregular, use_strict
	INTEGER	      dset, cdfid, status

* local parameter definitions:
	LOGICAL	    do_warn
	INTEGER     str_eq
	PARAMETER ( str_eq = 0,
     .		    do_warn = .TRUE. )

* local variable declarations:
*  ivtm - TMAP library variable ID
*  ivcd - netCDF file variable ID
      LOGICAL NC_GET_ATTRIB, 
     .        TM_LEGAL_NAME, MATCH_NAME,
     .        got_it, got_bad, its_epic, 
     .        got_a_variable, failed_axis, 
     .        coordvar, has_gdef
      INTEGER STR_UPCASE, STR_CASE_BLIND_COMPARE, TM_LENSTR1, 
     .        TM_LENSTR, NCF_INQ_DS, STR_SAME,
     .        nvars,  ndims, ngatts, recdim, 
     .        cdfstat, las_pos, first_pos, iaxis,
     .        ivcd, ivtm, vartyp, nvdim, vdims(8), nvatts, i, vlen,
     .        istat, npts, epic_t2,
     .        nbvar, ivar, len_test, len_mchars, 
     .        attype, attlen, attoutflag, maxlen, iatt, 
     .        varid, all_outflag, dset_num, attid, bad_bndsid,
     .	      axcalt, axlagf

      REAL vals  ! potential numerical attributes
      CHARACTER*128 aname, line1d

* ?? how many bounds variables (bname) might there be ??

      INTEGER max_vnamelen
      PARAMETER (max_vnamelen = 128)
      CHARACTER    vname*128, name*16, buff*128
      CHARACTER*128 bname(30)
      REAL global_bad
      REAL dg_bad

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

* initialize: save the state of grid memory prior to any new definitions

* ... flag all axes as non-reversed as the default
	DO i = max_lines, line_ceiling
 	   reversed(i) = .FALSE.
 	ENDDO

!* ... how many grid spaces used before we start ?
!	status = TM_FIND_GRID_SLOT ( firstgrid )
!        lastgrid = firstgrid - 1
!	IF (status .NE. merr_ok) GOTO 5900
!
!* ... how many line spaces used before we start ?
!	status = TM_FIND_LINE_SLOT ( firstax )
!        lastax = firstax - 1
!	IF (status .NE. merr_ok) GOTO 5900

* get the vital statistics for the data set 

        dset_num = dset
        IF (dset_num .LT. -2) dset_num = -2
        istat = NCF_INQ_DS (dset_num, ndims, nvars, ngatts, recdim )

        its_epic = ds_type(dset) .EQ. 'ECDF'

* set the  default value of the global_bad flag (used by EPIC but sometimes missing)
        IF ( its_epic ) THEN
           global_bad = 1.E35     ! default for EPIC
        ELSE
           global_bad = -1.E34    ! default for FERRET
        ENDIF

* See if there is a grid_definition dimension in the file
* If so will skip true-monthly axis computation in get-axis step.

	DO i = 1, ndims
           CALL CD_GET_DS_DIMS (dset, i, aname, npts, status)
	   has_gdef = (STR_SAME(aname, 'grid_definition') .EQ. 0) 
	   IF (has_gdef) EXIT  ! from loop
	ENDDO

* See if there is a global attribute missing_value in the file

        IF (ngatts .GT. 0) THEN
           CALL CD_GET_VAR_ID (dset, '.', varid, status)


* loop through global attributes looking for missing-value attrib
           DO 20 iatt = 1, ngatts

* get global attribute name
              name = ' '
              CALL CD_GET_VAR_ID (dset, '.', varid, status)
              CALL CD_GET_VAR_ATT_NAME( dset, varid, iatt, name, status)
              maxlen = 128
              IF ( name .EQ. 'missing_value' .OR. 
     .             name .EQ. 'MISSING_VALUE')
     .           got_bad = NC_GET_ATTRIB ( dset, varid, name, do_warn, 
     .                       '.', maxlen, attlen, attoutflag, 
     .                       buff, dg_bad)

              IF (got_bad) THEN
                 global_bad = dg_bad
              ENDIF
  20       CONTINUE

        ENDIF

*   * * * loop through all variables in the net CDF file * * *
*  Get any bounds attribute for each axis, save the bounds
*  name.  Then these names do not need to be processed as dependent 
*  variables.

        nbvar = 0
        DO 2000 ivcd = 1,nvars

* get the vital statistics for this variable

           CALL CD_GET_VAR_INFO (dset, ivcd, vname, vartyp,
     .                 nvdim, vdims, nvatts, coordvar, 
     .                 all_outflag, status)
           IF ( status .NE. merr_ok ) GOTO 5200
           vlen = TM_LENSTR1(vname)

* If it is marked as a coordinate variable but is of string type, then
* issue a message. The variable of the same name will contain the strings.

	   IF (coordvar .AND. (vartyp.EQ.NF_CHAR .OR. 
     .         vartyp.LT.NF_BYTE .OR. vartyp.GT.NF_DOUBLE) ) THEN 
	      CALL TM_NOTE(
     .           'Coordinate variable '//
     .           vname(:vlen)//' is of non-numeric type.', lunit_errors )
              status = STR_UPCASE( buff, vname )
              CALL TM_NOTE(
     .           'A dummy axis of subscripts will be used. '// 
     .           'Variable '//buff(:vlen)//' contains the strings', 
     .           lunit_errors )
              GOTO 400
	   ELSE IF (vartyp.LT.nf_byte .OR. vartyp.GT.nf_string) THEN
	      GOTO 5300
	   ENDIF
	    
*** DECIDE IF IT IS AN AXIS OR A DEPENDENT VARIABLE
* if it is not 1D it must be a dependent variable (3/99)

	   IF ( nvdim .NE. 1 ) GOTO 400

* if var name doesn't match a dim name it isn't a coordinate var
           CALL CD_GET_DIM_ID (dset, vname(:vlen), i)
           IF (i .EQ. atom_not_found) GOTO 400

* See if it has a bounds attribute 

           maxlen = 128
           got_it = NC_GET_ATTRIB( dset, ivcd, 'bounds',
     .              do_warn, vname(:vlen), maxlen, attlen, attoutflag, 
     .              bname(nbvar+1), vals )

           IF (got_it ) THEN

              CALL CD_GET_VAR_ATT_ID (dset, ivcd, 'bounds', attid, 
     .              status)
              IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, ivcd, 
     .              attid, aname, attype, attlen, attoutflag, istat )

              IF (attype .EQ. nf_char) THEN
                 nbvar = nbvar + 1
                 istat = STR_UPCASE( bname(nbvar), bname(nbvar) )
              ENDIF
           ENDIF

  400      CONTINUE
 2000   CONTINUE

*   * * * loop through all variables in the net CDF file * * *

* classify each as a coordinate variable (axis) or as a dependent
* variable. If a 1-D coordinate variable then TRY to interpret it,
* (in cd_get_1_axis) but if it proves to be invalid, then process
* it, instead, as a dependent variable
	las_pos  = 1    ! reduces redundant search for space
	got_a_variable = .FALSE.
	
	DO 1000 ivcd = 1,nvars

* initialize
	   failed_axis = .FALSE.

* get the vital statistics for this variable

           CALL CD_GET_VAR_INFO (dset, ivcd, vname, vartyp, nvdim,
     .                 vdims, nvatts, coordvar, all_outflag, status)
           IF ( status .NE. merr_ok ) GOTO 5200
           vlen = TM_LENSTR1(vname)

*** CHECK A NUMBER OF REASONS TO IGNORE THIS VARIABLE
* is the dimension name "grid_definition" ? If so, handled elsewhere

           CALL CD_GET_DS_DIMS (dset, vdims(1), buff, npts, istat)
           IF ( istat .NE. merr_ok ) GOTO 5200

           istat = STR_CASE_BLIND_COMPARE('grid_definition', buff(:15) )
           IF ( istat .EQ. str_eq ) GOTO 1000

* or is it milliseconds for an an EPIC time axis? If so, handled with "time"
           IF ( its_epic .AND. vname(:vlen).EQ.'time2' ) GOTO 1000

* Is it an edge definition ? If so, process w/ corresponding axis
           maxlen = max_vnamelen 
           got_it = NC_GET_ATTRIB( dset, ivcd, 'edges', do_warn, 
     .                            vname(:vlen), maxlen, attlen,
     .                            attoutflag, buff, vals)

           IF ( got_it .AND. buff.EQ.' ' ) GOTO 1000

* or is it really a child axis (avail for. generic CDF apps but not needed) ?
           maxlen = 16  !linenamlen
           got_it = NC_GET_ATTRIB( dset, ivcd, 'child_axis',
     .                            do_warn, vname(:vlen), maxlen, attlen,
     .                            attoutflag, name, vals ) 
           IF ( got_it ) GOTO 1000

* or is it really a bounds variable for an axis?  If so it is processed with the axis.

           len_test = TM_LENSTR(vname)
           IF (nbvar .GT. 0) then 
              DO 30 ivar = 1, nbvar
                 len_mchars = TM_LENSTR(bname(ivar))
                 IF (MATCH_NAME(vname, len_test, bname(ivar), len_mchars )) GOTO 1000
   30         CONTINUE
           ENDIF

* is it a supported dimensionality (6d, strings have extra dimension) ?
          IF ( nvdim .GT. nferdims+1 ) THEN
             CALL TM_NOTE('Unsupported netCDF # of axes for variable. Ignoring '
     .             //vname(:vlen), tt )
             GOTO 1000
          ENDIF

* is it a supported dimensionality (numeric can be 6D)?
          IF ( nvdim .GT. nferdims .AND. vartyp.NE. NCCHAR ) THEN
             CALL TM_NOTE('Unsupported netCDF # of axes for variable. Ignoring '
     .             //vname(:vlen), tt )
             GOTO 1000
          ENDIF

*** DECIDE IF IT IS AN AXIS OR A DEPENDENT VARIABLE
* if it is not 1D it must be a dependent variable (3/99)
	   IF ( nvdim .NE. 1 ) GOTO 500

* if for some reason it is desirable for an axis name not to match its dim
* name then the "parent_axis" attribute can flag that it is a parent axis 
           maxlen = 16  !linenamlen
           got_it = NC_GET_ATTRIB( dset, ivcd, 'parent_axis',
     .                            do_warn, vname(:vlen), maxlen,
     .                            attlen, attoutflag, name, vals )

           IF ( got_it ) GOTO 100

* if var name doesn't match a dim name it isn't a coordinate var
           CALL CD_GET_DIM_ID (dset, vname(:vlen), i)
           IF (i .EQ. atom_not_found) GOTO 500

* PROCESS AN AXIS	   
 100        CALL CD_GET_1_AXIS(dset, cdfid, ivcd, iaxis,
     .                         its_EPIC, reversed, tregular, use_strict, 
     .                         bad_bndsid, has_gdef, status)
           IF ( status .NE. merr_ok ) GOTO 5900

* invalid bounds or edges found. Make the bounds a dependent variable
	   IF (bad_bndsid .GT. 0) THEN
	     CALL CD_GET_1_DEP_VAR ( dset, cdfid, bad_bndsid, las_pos, global_bad, 
     .				   ivtm, status )
             GOTO 501
           ENDIF

	   IF (iaxis .NE. 0) GOTO 1000		! iaxis>0 if axis was made
	   failed_axis = .TRUE.			! go on and treat as a dep. var

* PROCESS A DEPENDENT VARIANLE
 500	   CALL CD_GET_1_DEP_VAR ( dset, cdfid, ivcd, las_pos, global_bad, 
     .				   ivtm, status )

 501       IF (status .NE. merr_ok) GOTO 5900
	   IF ( ivtm .NE. 0 ) THEN
	      las_pos = ivtm	! optimizes next hunt for slot
	      IF (.NOT.got_a_variable) THEN
	         first_pos = ivtm
	         got_a_variable = .TRUE.
	      ENDIF
	   ENDIF
	   IF ( failed_axis ) ds_var_titl_mod(ivtm) = 
     .				'invalid coordinate axis'
           
	   IF (bad_bndsid .GT. 0) THEN
	      ds_var_titl_mod(ivtm) = 'invalid axis bounds'
	      bad_bndsid = 0
	   ENDIF

* EPIC KLUDGE: if an EPIC file has an invalid "TIME" axis we want, not only
* "TIME" as a dependent variable, but also "TIME2"
	   IF ( its_epic .AND. vname(:vlen).EQ.'time' ) THEN
	      CALL CD_GET_VAR_ID ( dset, 'time2', epic_t2, status )
	      IF ( epic_t2 .GT. 0 ) THEN
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
* ... *kob* v541 8/02 - need to also check and make sure we are still dealing with the
*     same dataset.  if not, when variables from different data sets are interleaved 
*     in ds_var_code this can cause Ferret to not properly upcase variables from 
*     one dset that have the same name as variables in another dset. 
	      IF ( i .EQ. ivtm .OR. ds_var_setnum(i) .NE. dset) GOTO 300
	      istat = STR_CASE_BLIND_COMPARE(ds_var_code(ivtm),
     .                                       ds_var_code(i)    )
	      IF ( istat .EQ. str_eq ) GOTO 310                  ! leave as-is
 300	    CONTINUE
* ... (finally) upcase the name
*	  istat = STR_UPCASE( ds_var_code(ivtm), ds_var_code(ivtm) )
          CALL string_array_modify_upcase(ds_var_code_head,ivtm,
     .                                    ds_var_code(ivtm),
     .                                    LEN(ds_var_code(ivtm)))
 310    CONTINUE
	ENDIF

* successful completion
	status = merr_ok
	RETURN

* ERROR MESSAGES
 5200   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_SCAN_VARS', cdfid,
     .        no_varid, no_errstring, no_errstring, *5900 )
 5300   CALL TM_ERRMSG
     .     ( merr_notsupport, status, 'CD_SCAN_VARS', dset_num,
     .         no_stepfile, 
     .         'String Data Type not implemented: variable '//vname(:vlen),
     .         no_errstring,*5900)

* error exit
 5900   RETURN
	END
