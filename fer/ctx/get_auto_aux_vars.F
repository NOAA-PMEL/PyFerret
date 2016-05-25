	SUBROUTINE GET_AUTO_AUX_VARS( var, cat, dataset,
     .			              curv_vars, curv_cats )

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
* determine if curvilinear coordinates are applicable on a particular variable
* by looking for the __LayerzRef_ and __CurviRef_ attributes
* These may have been set either automatically, based upon CF 'coordinates'
* and 'formula_terms' attributes found in a netCDF file, or have been set
* manually using the SET VARIABLE/LAYERZ and /CURV commands

* inputs: var, cat, dset -- describe the variable for which we are seeking
*		auxiliary coordinates
* outputs: curv_vars, curv_cats -- the identities of the aux variables

* *sh* 12/13
* V693+ 11/14 *sh* renaming 'sigma' as 'layerz' throughout
* V7    5/16 *acm* See ticket 2352. For LET/D variables look for info under that dataset

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include 'xcommand.cmn'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xdset_info.cmn_text'

* calling argument declarations
	INTEGER	 var, cat, dataset,
     .		 curv_vars(nferdims), curv_cats(nferdims)

* internal variable declarations
	LOGICAL	  NC_GET_ATTRIB, got_it
	INTEGER	  TM_LENSTR1, 
     .		  idim, dset, attlen, attoutflag, maxlen, varid, slen,
     .		  status, aux_var, aux_cat
	REAL	  attvals(10)   ! only if __LayerzRef_ mis-defined as numeric!
	CHARACTER VAR_CODE*128, refvarname*128, varname*128,
     .		  dsetname*128

* XXXXX TEMPORARY -- if this is a uvar need to look at inherited uvar_aux*
* as sources of curvilinear aux infor BUT will they be defined  get_grid 

* initialize
	dset = dataset
	DO idim = 1, nferdims
	   curv_vars(idim) = unspecified_int4
	   curv_cats(idim) = unspecified_int4
	ENDDO
	varname = VAR_CODE(cat, var)

* user-defined variables may be associated with inherited auxiliary variables
	IF (cat .EQ. cat_user_var) THEN
	   DO idim = 1, nferdims
	      IF (uvar_aux_stat(var,idim) .EQ. paux_stat_passed
     .	     .OR. uvar_aux_stat(var,idim) .EQ. paux_stat_needed ) THEN
	         curv_vars(idim) = uvar_aux_var(var, idim, dataset)
	         curv_cats(idim) = uvar_aux_cat(var, idim, dataset)
	      ENDIF
	   ENDDO
	ENDIF

* get the variable ID in the linked list  (user vars stored under dset=pdset_uvars)
	IF (cat .EQ. cat_user_var) THEN
	   dset = pdset_uvars  ! signals a uvar
	   CALL CD_GET_VAR_ID (dset, varname, varid, status) 

* Check whether its a LET/D variable, stored with its dataset.
	   IF (dset.EQ.pdset_uvars .AND. status.NE.ferr_ok) THEN
	      IF (dataset .GT. pdset_irrelevant) 
     .      CALL CD_GET_VAR_ID (dataset, varname, varid, status)
              IF (status .EQ. ferr_ok) dset = dataset
	   ENDIF
	   
	ELSE
	   CALL CD_GET_VAR_ID (dset, varname, varid, status) 
	ENDIF

* get the attribute value of '__LayerzRef_' (patnam_layerz)
        maxlen = LEN(refvarname)
        got_it = NC_GET_ATTRIB ( dset, varid, patnam_layerz,
     .                           .FALSE., varname, maxlen,
     .                           attlen, attoutflag, refvarname,
     .                           attvals )
        IF (.NOT.got_it) RETURN

* Find the variable that has been named
* NOTE:  wired to do only a Z axis  at this time  XXXXXXXXXXX
        CALL FIND_VAR_NAME( dataset,
     .			    refvarname,
     .			    aux_cat, aux_var )
	IF (aux_var .EQ. munknown_var_name) THEN
* .... "Variable TEMP associated with non-existent layerz variable DEPTH"
*      "in levitus_climatology"
	   CALL GET_SHORT_DSET_NAME(dataset, dsetname, slen )
	   CALL WARN('Variable '//varname(:TM_LENSTR1(varname)) //
     .		' associated with non-existent layerz variable ' //
     .		refvarname(:TM_LENSTR1(refvarname)) // pCR //
     .		'           in ' //dsetname(:slen) )
	ELSE
	   curv_vars(z_dim) = aux_var
	   curv_cats(z_dim) = aux_cat
	ENDIF

 5000	RETURN

* error exits

	END
