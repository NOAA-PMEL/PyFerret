    	SUBROUTINE PROMOTE_AGG_GLOBAL_ATTS(agg_dset, nsets, member_sets, 
     .               agg_dim, iline, nvars, status)

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
*  in any advertising or publicity to endorse or promoted any products
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
* assign global attributes to the Ensemble aggregation 
* Promote the attribute to a variable on the aggregate-axis. 
* programmer -Ansley Manke from set_agg_var_atts
* NOAA/PMEL, Seattle, WA - Science Data Integration Group

* V741 *acm* 5/18 
* Notes
* 1. For now, promoting only string valued attributes.  For numeric
* ones this follows easily, but for multi-valued would need to get
* the max length and make another axis?  Or turn them into strings?
* 2. Do the same for forecast sets?  What about ensembles of Forecasts
* and FMRC's of ensembles?  Any issues?

        INCLUDE 'tmap_dims.parm'
#include "tmap_dset.parm"
        INCLUDE 'ferret.parm'
        INCLUDE 'errmsg.parm'      
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'xvariables.cmn'
        INCLUDE 'xcontext.cmn'
        INCLUDE 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xdset_info.cd_equiv'
	include 'xtm_grid.cmn_text' 
        INCLUDE 'netcdf.inc'

* calling arguments
*  agg_dset     - dset no of the aggregation (input)
*  nsets        - number of member datasets in aggregation (input)
*  member_sets  - list of member dataset numbers (input)
*  agg_dim      - direction of the aggregatin, just for a note
*  status       - return status (output)

* calling argument declarations
	INTEGER agg_dset, nsets, member_sets(*), agg_dim,  
     .         iline, nvars, status

* Local variable declarations
        INTEGER maxlen
        PARAMETER (maxlen = 2048)

        LOGICAL NC_GET_ATTRIB, TM_LEGAL_NAME, got_it, first_memb
        INTEGER TM_LENSTR1, STR_SAME, TM_FIND_LIKE_GRID,
     .          vlen, iref, rmemb, varid, ngatts, 
     .          imemb, attype, attlen, attoutflag, m_attid, m_attype, 
     .          m_attlen, m_attoutflag, i, ii, i1, ia, j, iatt, idim, 
     .          iset, igrid, ngrid, nv, ivar, igrid_dummy, vtype, 
     .          vtype_dummy, num_indices, slen, itype

        REAL    attvals(100)
	CHARACTER TM_FMT*32, 
     .          attname*128, vname*1, buff*128, label*8, bigstring*2048
        CHARACTER*(maxlen) attstr

* initialize

*     ... assign a label to use later 

* Promote global attributes from Ensembles. 
* Currently do not promote global attributes from members of an FMRC aggregation

        IF (agg_dim .EQ. e_dim) THEN
	   label = 'Ensemble'
c        ELSE IF (agg_dim .EQ. f_dim) THEN
c	   label = 'FMRC'
        ELSE
	   RETURN
	ENDIF

* Define the grid for the promoted variables

	igrid = 0
	CALL TM_ALLO_TMP_GRID (igrid, status)
	IF (status .NE. ferr_ok) GOTO 5000
	DO idim = 1, nferdims
	   grid_line(idim, igrid) = mnormal
	ENDDO
	grid_line(agg_dim, igrid) = iline

* dup names resolved later
	IF (agg_dim .EQ. e_dim) grid_name(igrid) = 'GGE'
	IF (agg_dim .EQ. f_dim) grid_name(igrid) = 'GGF'

* Define variables in the aggregation from the member-dataset global attributes.

* varid for global attributes is 0
        vname = '.'
	vlen = 1
	varid = 0
	first_memb = .TRUE.

* Loop over member sets. Make a variable which is the member-set names
* and variables with all global attributes in the collection, including
* ones present only in some member sets. iref is the current reference set, 
* checking if it has attributes that others, so far, haven't included.

	memberloop: DO iref = 1, nsets

	IF (member_sets(iref) .EQ. pdset_dummy) CYCLE memberloop
	rmemb = member_sets(iref)

* Get number of global attributes from the reference set
        CALL CD_GET_VAR_NATTS (rmemb, varid, vname(:vlen),
     .                         ngatts, status)

* First put in a promoted varaible that's the dataset name
	ia = 0
	IF (.NOT.first_memb) ia = 1

**** loop through the attribute list from the reference variable
        attloop: DO iatt = ia, ngatts

* Add a variable to the global set to contain the short-names of the members
* (store the name in attname)

	   IF (iatt .EQ. 0) THEN 

	      attname = 'member_dset'
	      attype = NCCHAR

	   ELSE

* All global attributes

              CALL CD_GET_VAR_ATT_NAME (rmemb, varid, iatt, attname, status)

* get the reference attribute info
              CALL CD_GET_VAR_ATT_INFO (rmemb, varid, iatt, attname, 
     .               attype, attlen, attoutflag, status )
              IF (status .NE. ferr_ok) GOTO 5200

* If this is not a legal name, skip it.  (e.g. attribute 'aggregate name')

              IF (.NOT.TM_LEGAL_NAME(attname) ) CYCLE attloop 

* Not doing numeric attributes, at least for now.

              IF ( attype .NE. NCCHAR ) CYCLE attloop 

* get the reference attribute value and info
              got_it = NC_GET_ATTRIB ( rmemb, varid, attname,
     .                              .TRUE., vname(:vlen), maxlen,
     .                              attlen, attoutflag,
     .                              attstr, attvals )
              IF (.NOT.got_it) GOTO 5200

	   ENDIF  ! iatt .EQ. 0

	   first_memb = .FALSE.

* Find the global attribute in other sets, check for type mismatch
* and check that we can get the attribute.  In each pass treating each
* member file as iref - look for attributes not yet found


           DO imemb = 1, nsets
	   
	      IF (imemb .EQ. iref) CYCLE

* ... skip over dummy members
              IF (member_sets(imemb) .EQ. pdset_dummy) CYCLE
	      iset = member_sets(imemb)

* ... iatt = 0 getting dataset names

	      IF (iatt .EQ. 0) THEN 

	         attname = 'member_dset'

	      ELSE	

* ... get the attribute info for the member

                 CALL  CD_GET_VAR_ATT_ID (iset, varid, attname, 
     .                                m_attid, status)

                 IF (status .NE. ferr_ok) CYCLE 

* Have we already defined a promoted variable from this attribute?
	         IF (iref .GT. 1) THEN
	            CALL string_array_find_caseblind(ds_var_code_head,
     .                                   attname,
     .                                   LEN(attname),
     .                                   string_array_result(1),
     .                                   max_uvar,
     .                                   num_indices)

	            IF (num_indices .GE. 1) THEN
                       DO j = 1, num_indices
                          i1 = string_array_result(j)
* ... We already have promoted this attr. Move to next attr in set # iref 
                          IF (ds_var_setnum(i1) .EQ. agg_dset) CYCLE attloop
                       ENDDO
	            ENDIF
	         ENDIF

	         CALL CD_GET_VAR_ATT_INFO (iset, varid,
     .                                  m_attid, attname, m_attype, 
     .                                  m_attlen, m_attoutflag, status )
		
* is one numeric and the other char? Skip the attrib.

                 IF (attype.EQ.NCCHAR .AND. m_attype.NE.NCCHAR ) CYCLE attloop
                 IF (attype.NE.NCCHAR .AND. m_attype.EQ.NCCHAR ) CYCLE attloop

* check that we can get the attribute value for the member
                 got_it = NC_GET_ATTRIB (iset, varid,
     .                              attname,
     .                              .TRUE., vname(:vlen), maxlen,
     .                              m_attlen, m_attoutflag,
     .                              attstr, attvals )

                 IF (.NOT. got_it) CYCLE 

              ENDIF ! iatt.EQ.0 getting dataset names

           ENDDO  ! Getting attrib info from all sets


* Create a variable in the aggregate dataset 
* ds_var slot, fill with aggregate variable in aggregate dataset.

           DO nv = 1, maxvars
	      IF ( STR_SAME(ds_var_code(nv),char_init) .EQ. 0 ) EXIT
           ENDDO

           IF (iatt.EQ.0) THEN
	      buff = "Member dataset names in "//label(:TM_LENSTR1(label))

           ELSE

              buff = "Global Att *"//attname(:TM_LENSTR1(attname))//"* in "//
     .		label(:TM_LENSTR1(label))

* ... check whether there are file variables with the same name as the global attr
              dsetcheck: DO ii = 1, nsets
                 iset = member_sets(ii)

                 IF (iset .EQ. pdset_dummy) CYCLE dsetcheck
* ... check whether there are file variables with the same name as the global attr
                 varcheck: DO ivar = 1, maxvars
                    IF ( (ds_var_setnum(ivar) .EQ. iset) .AND. 
     .                   (STR_SAME(ds_var_code(ivar), attname) .EQ. 0) ) THEN
                       CYCLE attloop
	            ENDIF
                 ENDDO varcheck
              ENDDO dsetcheck

	   ENDIF

	   ds_var_title(nv) =  buff

* Now add a user-defined variable to the aggregated dset 

           ds_var_setnum(nv)   = agg_dset
           CALL string_array_modify_upcase(ds_var_code_head, nv, 
     .           attname, LEN(attname))

           ds_var_titl_mod(nv) = " "
           ds_var_units(nv)    = " "
           ds_missing_flag(nv) = bad_val4
           ds_bad_flag(nv)     =bad_val4
           IF (attype .EQ. NCCHAR) THEN 
              ds_var_type(nv)  = ptype_string
           ELSE
              ds_var_type(nv)  = ptype_float
	   ENDIF
	   cd_scale_factor(nv) = 1.
	   cd_add_offset(nv)   = 0.
	   cd_scaled(nv)       = .FALSE.

	   DO idim = 1, nferdims
              ds_grid_start(idim,nv)= 1
              ds_grid_end(idim,nv)  = 1
              ds_ordering(idim,nv)  = int4_init
	   ENDDO
	   ds_grid_start(agg_dim,nv)= 1
	   ds_grid_end(agg_dim,nv)  = nsets
	   ds_ordering(1,nv)        = agg_dim  ! last dimension of the grid

	   ds_grid_number(nv) = igrid

* Hook the aggregated variable into the linked list structures
	      
	   nvars = nvars + 1
	   CALL CD_ADD_VAR( agg_dset, attname, nvars, 
     .            ds_var_title(nv), ds_var_units(nv), attype, 
     .            .FALSE., ds_bad_flag(nv), status)

	   DO imemb = 1, nsets
              iset = member_sets(imemb)
              IF (iset .EQ. pdset_dummy) CYCLE
              itype = cat_agg_gpromote
              CALL CD_ADD_AGG_VAR_INFO (agg_dset, attname, imemb, itype, 
     .             iset, igrid, iline, nv, status)
	   ENDDO

        ENDDO attloop

        ENDDO memberloop

* success
* (status may have been set to atom_not_found on a missing attribute.
*  Set to ferr_ok.)

	status = ferr_ok
 5000	CONTINUE
c        DEALLOCATE(inote)
	RETURN

 5200   CALL ERRMSG (ferr_internal, status,
     .               'attListCrptn: '//attname(:vlen),*5000)
	END
