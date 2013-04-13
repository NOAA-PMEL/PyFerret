	SUBROUTINE CD_NF_GET_VAR_GRID ( dset, vname, gridlines, status )

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
* Just do an nf_get_varid for a variable that has been defined with LET/REMOTE.
* Call this after the dataset has been redefined with the encoded variable
* definitions.
*
* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Thermal Modeling and Analysis Program
* 2/2013  Ferret v6.85
*
* arguments:
*	dset	- data set number 

#include "netcdf.inc"
        include 'tmap_errors.parm'
	include 'tmap_dims.parm'
	include 'xstep_files.cmn_text'
	external xstep_files_data
        include 'xtm_grid.cmn_text'


* Argument variable declarations
	INTEGER   dset, gridlines(*), status
	CHARACTER*(*) vname

* Local variable declarations
	INTEGER STR_SAME, TM_LENSTR1, 
     .		cdfid, varid, ndims, dimids(nferdims),
     .		idim, igrid, dimlen, outgrid(nferdims), slen
	CHARACTER*500 buff, errbuff

	cdfid = sf_lunit(dset)

#ifdef usingDODSf2cUnderscore 
	status = NF_INQ_VARID_(cdfid, vname, varid) 
	status = NF_INQ_VARNDIMS_(cdfid, varid, ndims) 
	status = NF_INQ_VARDIMID _(cdfid, varid, dimids) 
#else
	status = NF_INQ_VARID(cdfid, vname, varid)
	status = NF_INQ_VARNDIMS(cdfid, varid, ndims)
	status = NF_INQ_VARDIMID (cdfid, varid, dimids)
#endif
	
	IF ( status .NE. NF_NOERR ) GOTO 5100

	DO 100 igrid = 1, nferdims
	   outgrid(igrid) = 0
  100	CONTINUE

	DO 300 idim = 1, ndims
#ifdef usingDODSf2cUnderscore 
	   status = NF_INQ_DIMNAME_(cdfid, dimids(idim), buff)
	   status = NF_INQ_DIMLEN_ (cdfid, dimids(idim), dimlen)
#else
	   status = NF_INQ_DIMNAME (cdfid, dimids(idim), buff)
	   status = NF_INQ_DIMLEN  (cdfid, dimids(idim), dimlen)
#endif
	   DO 200 igrid = 1, nferdims
	      IF (STR_SAME ( buff, line_name_orig(gridlines(igrid)) ) .EQ. 0) THEN
     	        outgrid(igrid) = gridlines(igrid)
	      ENDIF
  200	   CONTINUE
  300	CONTINUE

	DO 400 igrid = 1, nferdims
	   gridlines(igrid) = outgrid(igrid)
  400	CONTINUE

* successful completion
        status = merr_ok
 5000	RETURN

* error messages
 5100   CONTINUE

        CALL CD_TRANSLATE_ERROR (status, errbuff)
	slen = TM_LENSTR1(errbuff)

        buff = 'Unable to get varid for LET/REMOTE variable: '//
     .    errbuff(:slen)

        CALL TM_ERRMSG
     .     ( merr_badfileatt, status, 'CD_NF_GET_VAR_GRID', dset,
     .        no_stepfile, buff, no_errstring, *5000 )
	END
