	SUBROUTINE CD_ASSOC_GRIDS (dset, reversed, 
     .				   temp_axnams, status)


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
* Creates the association of the FERRET variables with their parent grids
* within a netCDF file
* Checks that all variables in the data set have associated grids.

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* 2/95 - added support for reverse-ordered coordinate axes
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef
* V500:  4/99 *kob* - increase size of temp_axnams and vname from 16 to 64
* V510 *sh* 3/00 - convert to use of dynamic grids and axes
* V533 *sh* 6/01 - support for string variables
* V552:  3/03 *acm* - increase size of vname to 128
* V600 *ACM* attribute control. 
*             NCVINQ replaced by CD_GET_VAR_INFO
*             NCDINQ replaced by CD_GET_DS_DIMS
*             Change call to CD_GET_ATTRIB to NC_GET_ATTRIB
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* TODO: initially leave this as 4D -- what to do with old files that have 4 slab min/max?
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* argument definitions:
*       dset        - TMAP data set pointer
*       reversed    - logical array to record which axes have reversed coords
*       temp_axnams - temporary storage of axis names to construct grids
*	status	    - return status

* include files

        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	        include 'xdset_info.cd_equiv'

* argument declarations
	INTEGER	 dset, status
        LOGICAL  reversed(max_lines:line_ceiling)
* 4/99 *kob* 
	CHARACTER*128 temp_axnams(nferdims,max_temp_grid)

* local parameter definitions:
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )

* local variable declarations:
* move do_warn assignment to paramater for linux *kob*
      LOGICAL NC_GET_ATTRIB,
     .        got_it, do_warn
      PARAMETER (do_warn = .TRUE.)
      INTEGER STR_CASE_BLIND_COMPARE,
     .        igrid, iaxis, istat, tmvar, ivar, ndim, idim,
     .        maxlen, attlen, attoutflag, all_outflag
      REAL    tmp
      CHARACTER*16 gname
* variables for final (unnecessary?) check
      CHARACTER TM_STRING*13, vname*128, buff*128
      INTEGER TM_LENSTR1,
     .        vartyp, nvdim, vdims(8), nvatts, dim, npts, vlen
      LOGICAL coordvar

* *kob* explicit definitions not allowed on decl line in linux...moved it to parameter line
      CHARACTER*1     dimnam(nferdims)
      DATA dimnam/'I', 'J', 'K', 'L', 'M', 'N'/

* loop through each variable in this TMAP data set
* if it has a "parent_grid" then associate it with that grid
* if not then make sure it has a grid associated

        DO 1000 tmvar = 1,maxvars
           IF ( ds_var_setnum(tmvar) .NE. dset ) GOTO 1000
           ivar = cd_varid(tmvar)

* does it point explicitly to a parent grid ?
          maxlen = 16  !gridnamlen
          got_it = NC_GET_ATTRIB( dset, ivar, 'parent_grid',
     .                         .NOT.do_warn, ' ', maxlen, attlen,
     .                         attoutflag, gname, tmp )  
          IF ( .NOT.got_it ) THEN
* ... check that it is a current temporary grid -- should ALWAYS be true
	     IF (ds_grid_number(tmvar) .LT. max_grids    ) GOTO 5100
	     IF (tmp_nam_ptr(ds_grid_number(tmvar)).EQ.0 ) GOTO 5100
             GOTO 1000
          ENDIF

* locate the grid it points to
	  igrid = 0	! to get start of linked list
 100	  CALL TM_NEXT_TMP_GRID( igrid, *5100)
             istat = STR_CASE_BLIND_COMPARE(gname, grid_name(igrid))
             IF ( istat .EQ. str_eq ) THEN
                ds_grid_number(tmvar) = igrid
                GOTO 110
             ENDIF
	  GOTO 100

* set ds_ordering if not explicitly given in file
* e.g. map X NORMAL Z T to 1 3 4 0
 110      IF ( ds_ordering(1,tmvar) .EQ. unspecified_int4 ) THEN
             DO 190 idim = 1, nferdims
               ds_ordering(idim,tmvar) = 0
 190         ENDDO
             ndim = 0
             DO 200 idim = 1, 4
                IF ( grid_line(idim,igrid) .EQ. mpsnorm ) GOTO 200
                ndim = ndim + 1
                ds_ordering(ndim,tmvar) = idim
 200         CONTINUE
          ENDIF

* count the number of axes in the grid
          ndim = 0
          DO 220 idim = 1, 4
             IF ( grid_line(idim,igrid).NE.mpsnorm ) ndim = ndim + 1
 220      CONTINUE
          
* set ds_grid_start if not explicitly given in file
          IF ( ds_grid_start(1,tmvar) .EQ. unspecified_int4 ) THEN
             DO 300 idim = 1, 4
               ds_grid_start(idim,tmvar) = 1
 300         CONTINUE
          ENDIF
	  
* set ds_grid_end if not explicitly given in file
          IF ( ds_grid_end(1,tmvar) .EQ. unspecified_int4 ) THEN
             DO 400 idim = 1, 4
                iaxis = grid_line(idim,igrid)
                IF ( iaxis .EQ. mpsnorm ) THEN
                   ds_grid_end(idim,tmvar) = 1
                ELSE
                   ds_grid_end(idim,tmvar) = line_dim(iaxis)
                ENDIF
 400         CONTINUE
          ENDIF

* TODO won't need this after implementing full 6D treatment here.
          DO 410 idim = 5, nferdims
             ds_grid_start(idim,tmvar) = 1
             ds_grid_end(idim,tmvar) = 1
 410      CONTINUE

* check that the parent lo/hi limits are consistent with the variable slab size

          CALL CD_GET_VAR_INFO (dset, ivar, vname, vartyp, nvdim, vdims,
     .                 nvatts, coordvar, all_outflag, istat)
          vlen = TM_LENSTR1(vname)

* ... if a string variable, the first dimension is the string axis (ignore it)
	  IF (vartyp .EQ. NF_CHAR) THEN
	     nvdim = nvdim - 1
	     DO idim = 1, nvdim
	        vdims(idim) = vdims(idim+1)
	     ENDDO
	  ENDIF
          IF ( ndim .NE. nvdim ) GOTO 5300 
          DO 500 idim = 1, 4
             dim = ds_ordering(idim,tmvar)
             IF ( dim .EQ. 0 ) GOTO 500
!             IF ( dim .EQ. unspecified_int4 ) dim = idim
             CALL CD_GET_DS_DIMS ( dset, vdims(idim), buff, npts, istat )
             IF ( npts .NE.
     .          ds_grid_end(dim,tmvar)-ds_grid_start(dim,tmvar)+1 )
     .               GOTO 5200
 500      CONTINUE

* flag (negative) the axes that have reverse-ordered coordinates
	  DO 610 idim = 1, 4
	     dim = ds_ordering(idim,tmvar)
             IF ( dim .EQ. 0 ) GOTO 610
* At this stage the axis pointers have not yet been inserted into the
* parent grid definitions (we still have to check for duplicate def'ns, etc.)
* So we need to check by name (among the axes defined in this data set)
* to find this axis and see if this axis is reversed.
	     IF ( grid_line(dim,igrid) .EQ. unspecified_int4 ) THEN
	       vname = temp_axnams(dim, tmp_nam_ptr(igrid))
	       iaxis = 0	! to get start of linked list
* ... search the linked list of axes defined in this dtaa set
 600	       CALL TM_NEXT_TMP_LINE( iaxis, *610)
                 IF ( vname .EQ. line_name(iaxis) ) THEN
	           IF (reversed(iaxis)) ds_ordering(idim,tmvar) = -1*dim
                   GOTO 610
                 ENDIF
	       GOTO 600
	    ENDIF
 610	  CONTINUE

 1000   CONTINUE

* successful completion
 	status = merr_ok
        RETURN

* error messages
 5100   CALL TM_ERRMSG(merr_unkgrid, status, 'CD_ASSOC_GRIDS',
     .       dset, no_stepfile,
     .       'CDF grid name: '//gname, no_errstring, *5900)

 5200   CALL TM_ERRMSG(merr_badsubscr, status, 'CD_ASSOC_GRIDS',
     .       dset, no_stepfile, 'CDF variable "'//vname(:vlen)//
     .       '" on '//dimnam(dim)//'='//
     .       TM_STRING(DBLE(ds_grid_start(dim,tmvar)))//'-'//
     .       TM_STRING(DBLE(ds_grid_end  (dim,tmvar))),
     .       no_errstring, *5900)

 5300   CALL TM_ERRMSG(merr_gridpredef, status, 'CD_ASSOC_GRIDS',
     .       dset, no_stepfile,
     .       '# of dims in variable '//vname(:vlen)//
     .       ' does not match grid '//gname, no_errstring, *5900)

* error exit
 5900   RETURN
	END
