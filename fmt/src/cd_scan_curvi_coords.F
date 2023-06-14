	SUBROUTINE CD_SCAN_CURVI_COORDS (dset, status)

*
* Look for 'coordinates' attributes on variables; report on auxiliary
* (i.e. curvillinear) coordinates on opening the dataset and on SHOW DATA

* Programmer Ansley Manke

* revision 0.0 - 5/23 for PyFerret issue 135

* argument definitions:
*       dset        - TMAP data set pointer
*	status	    - return status

* include files

        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'

	include 'xdset_info.cmn_text'
	include 'xdset_info.cd_equiv'
        include 'xio.cmn_text'

* argument declarations
	INTEGER	 dset, status


* local parameter definitions:
	LOGICAL	    do_warn
	INTEGER     str_eq, max_names
	PARAMETER (str_eq = 0)
	PARAMETER (do_warn = .TRUE.)
	PARAMETER (max_names = 100)  

* local variable declarations:
	LOGICAL NC_GET_ATTRIB, CD_ISIT_COORDVAR, got_it, coordvar

	INTEGER TM_LENSTR1, STR_SAME,
     .		i, j, tmvar, ivar, num_names,
     .		ndims, nvars, ngatts, iisame,
     .		maxlen, attlen, attoutflag, nnames
	CHARACTER aux_coordinates*500, vname*128, name*16,
     .		  dimnames(40)*128, aux_coord_names(8)*128, errmsg*160
	REAL vals  ! dummy, potential numerical attributes
	
* This routine sets the following flag; initialize to false.

	ds_has_aux_coords(dset) = .FALSE.

* Get names of the dataset dimensions. IF coordinate attributes 
* include dimensions we will skip over those in the list.
	
	CALL CD_GET_DS_INFO (dset, ndims, nvars, ngatts,
     .    nnames, status)

	DO i = 1, ndims
	   CALL CD_GET_DS_DIMS (dset, i, dimnames(i), j, status)
	ENDDO

* loop through each variable in this TMAP data set
* IF it has a "coordinates" attribute THEN associate it with that variable
* and flag that variable as an auxiliary coordinate

	maxlen = 500 ! length of aux_coordinates

	nnames = 0
        DO tmvar = 1,maxvars
           IF ( ds_var_setnum(tmvar) .NE. dset ) CYCLE
           ivar = cd_varid(tmvar)
	   call CD_GET_VAR_VARNAME (dset, ivar, vname, status)

	   coordvar = CD_ISIT_COORDVAR(dset, ivar, vname, status) 
	   IF (coordvar) CYCLE

	   got_it = NC_GET_ATTRIB ( dset, ivar, 'coordinates', do_warn, vname, 
     .			maxlen, attlen, attoutflag, aux_coordinates, vals )
           IF ( .NOT. got_it ) CYCLE

* parse the list of white-space-separated auxiliary variable names
           CALL TM_PARSE_STRING(aux_coordinates, max_names,
     .				   aux_coord_names, num_names)

* Are the coordinate variables names different from dimension names?

	   nnames = 0
	   DO i = 1, num_names
	      name = aux_coord_names(i)
	      iisame = 0
	      DO j = 1, ndims
	         iisame = STR_SAME ( name, dimnames(j) ) 
		 IF (iisame .EQ. 0) EXIT
	      ENDDO
	      IF (iisame .EQ. 0) CYCLE
	      
	      nnames = nnames + 1
	      aux_coord_names(nnames) = name
		    
* Is the coordinate variable in the dataset?  
* (Should this be an error? maybe a note?)

c	      CALL CD_GET_VAR_ID (dset, name, ivar, status)
c	      IF (status .NE. merr_ok) GOTO 5100
	   ENDDO
	   IF (nnames .GT. 0) ds_has_aux_coords(dset) = .TRUE.

	ENDDO

	IF (ds_has_aux_coords(dset)) 
     .   CALL TM_NOTE(
     .   '   Dataset variables have Auxiliary or Curvilinear Coordinates. Examine "coordinates" attributes.', 
     .   lunit_errors)

* successful completion
 	status = merr_ok
        RETURN

* error exit
 5100   errmsg = 'Variable '//vname(:tm_lenstr1(vname))//
     .   ' has coordinate attribute but coord variable '//
     .   name(:tm_lenstr1(name))//' not in the dataset'
	CALL TM_ERRMSG ( merr_badgriddef, status, 'CD_SCAN_CURVI_COORDS', dset,
     .        no_stepfile, errmsg, no_errstring, *5900 )
 5900   RETURN
	END
