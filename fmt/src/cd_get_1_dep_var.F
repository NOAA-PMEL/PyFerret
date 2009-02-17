	SUBROUTINE CD_GET_1_DEP_VAR ( dset, cdfid, ivcd, las_pos, global_bad, 
     .				      ivtm, status )

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
* Loads common blocks XDSET_INFO and XSTEP_FILES with vital system and
* background information for a single dependent variable from a netCDF file
* It processes FERRET special attributes without regard to whether variables
* are "generic" or "FERRET"

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500 3/99 *sh* -- new with code extracted from cd_scan_vars.F
*      4/99 *kob*: increase size of vname from 24 to 64 chars
* V533 6/01 *sh* -- set the variables data type
* V542 11/02*acm*  For 4-D string NetCDF files, allow nvdim to be 5 when
*                  code checks for "Unsupported netCDF # of axes".
* V552 5/03 *acm*: increase size of vname from 64 to 128 chars
* V600 *ACM* attribute control. 
*             Replace NCAINQ with CD_GET_VAR_ATT_INFO
*             Change call to CD_GET_ATTRIB to NC_GET_ATTRIB - get attrib from 
*               linked list structure
*             CD_GET_ATTVAL replaced by NC_GET_ATTRIB, which now returns  
*               strings or real values according to attrib type
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* V600  6/06 *acm*   CD_GET_VAR_ATT_INFO uses varid and attid rather than attname.
* V62   2/09 *acm* - allow for string-type missing and fill attributes to be read.

* argument definitions:
*       dset    - pointer to TMAP data set
*	cdfid   - id of currently open CDF file
*	ivcd	- variable id in netCDF file
*	las_pos  - TMAP variable pointer for last variable stored (for optimization, only - "1" OK) 
*	global_bad - a global bad value flag for all vars in dada set

*	ivtm	- returned TMAP variable pointer for the new variable (0 if unsuccessful)
*	status	- return status

* argument declarations:
	INTEGER	      dset, cdfid, ivcd, las_pos, ivtm, status
	REAL*4	      global_bad

* include files

        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xdset_info.cd_equiv'
        include 'xio.cmn_text'
        external xio_data


      LOGICAL NC_GET_ATTRIB_FLOAT, NC_GET_ATTRIB_STRING, 
     .        NC_GET_ATTRIB, CD_GET_BOUNDS,
     .        got_it, got_bad, got_missing, tf2,
     .        do_warn, coordvar
      PARAMETER (do_warn = .TRUE. )
      INTEGER TM_LENSTR1, TM_FERRET_VARTYPE, 
     .        cdfstat, idim,
     .        vartyp, nvdim, vdims(6), nvatts, vlen,
     .        recdim, istat, lasdim, attype, attlen, attoutflag, 
     .        maxlen, all_outflag, attid
      REAL    val
* 4/99 *kob* increase size of vname
      CHARACTER    vname*128, aname*128, buff*128
      CHARACTER*13 TM_STRING

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)


* special equivalence (cannot include in tmap_format/xdset_info.cd_equiv because "BYTE"
* isn't supported on DECstation
* reorder loop logic because linux cpp didn't like it *kob* 10/96


#ifdef NEED_BYTE
      BYTE cd_data_type(maxvars)
#else
      INTEGER*1 cd_data_type(maxvars)
#endif


      EQUIVALENCE ( ds_precision, cd_data_type )
      REAL    r4_grid_start(4,maxvars),
     .        r4_grid_end(4,maxvars)
      EQUIVALENCE ( r4_grid_end,   ds_grid_end ),
     .            ( r4_grid_start, ds_grid_start )

* get the vital statistics for this variable

           CALL CD_GET_VAR_INFO(dset, ivcd, vname, vartyp, 
     .                 nvdim, vdims, nvatts, coordvar, 
     .                 all_outflag, status)
           IF ( status .NE. merr_ok ) GOTO 5200
           vlen = TM_LENSTR1(vname)

* find the next storage position in TMAP common
	  DO 150 ivtm = las_pos,maxvars
	    IF (ds_var_setnum(ivtm) .EQ. set_not_open) GOTO 151
  150     CONTINUE
* oops - no room left for new variables
	  GOTO 5100

* set obvious defaults for this variable
 151	  cd_varid(ivtm)        = ivcd ! netCDF variable ID
          ds_var_setnum(ivtm  ) = dset
          ds_var_title(ivtm)    = ' '
          ds_var_titl_mod(ivtm) = ' '
          ds_var_units(ivtm)    = ' '
	  ds_missing_flag(ivtm)	= global_bad
	  ds_bad_flag(ivtm)	= ds_missing_flag(ivtm)
	  ds_precision(ivtm)	= 'SINGLE'
          cd_scale_factor(ivtm) = 1.0
          cd_add_offset(ivtm)   = 0.0
          cd_scaled(ivtm)       = .FALSE.
	  DO 160 idim = 1,4
             ds_grid_start(idim,ivtm)= unspecified_int4
             ds_grid_end  (idim,ivtm)= unspecified_int4
  	     ds_ordering(idim,ivtm)= unspecified_int4
 160      CONTINUE

* is it a supported data type ?
          cd_data_type(ivtm) = vartyp
	  ds_var_type(ivtm) = TM_FERRET_VARTYPE(vartyp)  ! 6/01

* is it a supported dimensionality ?
          IF ( nvdim .GT. 5 ) THEN
             CALL TM_NOTE('Unsupported netCDF # of axes for variable '
     .             //vname(:vlen), tt )
             GOTO 200
          ENDIF

* see what generic attributes can be read: long_name, units, missing_value

          IF (nvatts .GT. 0) THEN
             maxlen = 128
             got_it = NC_GET_ATTRIB_STRING( dset, ivcd, 'long_name',
     .                     .NOT.do_warn, vname(:vlen), maxlen, attlen, 
     .                     attoutflag, ds_var_title(ivtm))

             maxlen = 128
             got_it = NC_GET_ATTRIB_STRING( dset, ivcd, 'long_name_mod',
     .                     .NOT.do_warn, vname(:vlen), maxlen, attlen, 
     .                     attoutflag, ds_var_titl_mod(ivtm))

             maxlen = 64
             got_it = NC_GET_ATTRIB_STRING( dset, ivcd, 'units',
     .                     do_warn, vname(:vlen), maxlen, attlen, 
     .                     attoutflag, ds_var_units(ivtm))

             got_it = NC_GET_ATTRIB_FLOAT( dset, ivcd, 'scale_factor',
     .                     do_warn, vname(:vlen), attlen, 
     .                     attoutflag, cd_scale_factor(ivtm))

             tf2    = NC_GET_ATTRIB_FLOAT( dset, ivcd, 'add_offset',
     .                     do_warn, vname(:vlen), attlen, 
     .                     attoutflag, cd_add_offset(ivtm))

             cd_scaled(ivtm) = got_it .OR. tf2

* netCDF files may use "standard" attributes missing_value and/or
* _FillValue to flag bad data
* For the NOAA COOP standard the missing_value flag must be stored in the
* packed data type.  If so it must be unpacked.  (2/95)
	     
	     CALL CD_GET_VAR_ATT_ID (dset, ivcd, 'missing_value', 
     .                  attid, istat)
	     IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .                 ivcd, attid, aname, attype, attlen, attoutflag, 
     .                 istat)
     
             IF (attype .NE. NCCHAR) THEN
                got_bad = NC_GET_ATTRIB_FLOAT( dset, ivcd, 
     .                'missing_value',
     .                do_warn, vname(:vlen), attlen,
     .                attoutflag, ds_bad_flag(ivtm))
             ELSE
  	        maxlen = 128
                got_bad = NC_GET_ATTRIB( dset, ivcd, 'missing_value',
     .                do_warn, vname(:vlen), maxlen, attlen,
     .                attoutflag, buff,  ds_bad_flag(ivtm))
             ENDIF


	     
	     CALL CD_GET_VAR_ATT_ID (dset, ivcd, '_FillValue', 
     .                  attid, istat)
	     IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .                 ivcd, attid, aname, attype, attlen, attoutflag, 
     .                 istat)
     
             IF (attype .NE. NCCHAR) THEN
                got_missing = NC_GET_ATTRIB_FLOAT( dset, ivcd, 
     .                '_FillValue',
     .                do_warn, vname(:vlen), attlen,
     .                attoutflag, ds_missing_flag(ivtm))
             ELSE
  	        maxlen = 128
                got_missing = NC_GET_ATTRIB( dset, ivcd, '_FillValue',
     .                do_warn, vname(:vlen), maxlen, attlen,
     .                attoutflag, buff,  ds_bad_flag(ivtm))
             ENDIF

* ... need to unpack the missing_value flag?
*     (note that the netCDF library insists on _FillValue's data type)
             IF ( vartyp.LE.NCLONG .AND. cd_scaled(ivtm) ) THEN
                IF ( got_bad ) THEN
                   CALL CD_GET_VAR_ATT_ID (dset, ivcd, 'missing_value', 
     .                  attid, istat)
                   IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .                 ivcd, attid, aname, attype, attlen, attoutflag, 
     .                 istat)
	           IF ( attype .EQ. vartyp ) ds_bad_flag(ivtm) =
     .			ds_bad_flag(ivtm)*cd_scale_factor(ivtm)
     .			+ cd_add_offset(ivtm)
	        ENDIF

* ... *kob* Also need to see about unpacking _FillValue
                IF ( got_missing ) THEN
                   CALL CD_GET_VAR_ATT_ID (dset, ivcd, '_FillValue', 
     .                  attid, istat)
                   IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .                  ivcd, attid, aname, attype, attlen, attoutflag, 
     .                  istat )
	           IF ( attype .EQ. vartyp ) ds_missing_flag(ivtm) =
     .	   		ds_missing_flag(ivtm)*cd_scale_factor(ivtm)
     .			+ cd_add_offset(ivtm)
	        ENDIF
             ENDIF
             IF ( got_bad .AND. .NOT.got_missing )
     .           ds_missing_flag(ivtm) =  ds_bad_flag(ivtm)
             IF ( got_missing .AND. .NOT.got_bad )
     .           ds_bad_flag(ivtm) =  ds_missing_flag(ivtm)


*  *kob* call SWITCH_NAN to make sure if NaN exisits, its in ds_missing_flag
	     CALL SWITCH_NAN(ds_bad_flag(ivtm), ds_missing_flag(ivtm))

* see what special FERRET attributes can be read: axis ordering and
* sub-hyperslab bounds
             maxlen = 4
             got_it = NC_GET_ATTRIB( dset, ivcd, 'axis_order',
     .                do_warn, vname(:vlen), maxlen, attlen, 
     .                attoutflag, buff, val)
             lasdim = 0
             IF (got_it) THEN
                DO 170 idim = 1, 4
                   IF ( buff(idim:idim) .EQ. ' ' ) THEN
                     ds_ordering(idim,ivtm) = 0
                     GOTO 170
                  ELSEIF ( buff(idim:idim) .EQ. 'T' ) THEN
                     ds_ordering(idim,ivtm) = 4
                  ELSE
                     ds_ordering(idim,ivtm) = ICHAR(buff(idim:idim))-87 ! XYZ->123
                  ENDIF
                  IF ( ds_ordering(idim,ivtm) .LT. 0
     .            .OR. ds_ordering(idim,ivtm) .GT. 4 ) THEN 
                     CALL TM_NOTE(
     .                     'Illegal netCDF "axis_order" for variable '
     .                     //vname(:vlen), tt )
		     GOTO 200
                   ENDIF
                   IF ( ds_ordering(idim,ivtm) .LE. lasdim ) THEN
                      CALL TM_NOTE(
     .                     '"axis_order" permutations not supported: '
     .                        //vname(:vlen), tt )
                      GOTO 200
                   ENDIF
                   lasdim = ds_ordering(idim,ivtm)
 170            CONTINUE
             ENDIF

* does this variable have a FERRET "parent grid" ?
* if so check for explicit hyperslab limits
             buff = ' '   ! problems from characters beyond 16
             maxlen = 16 !gridnamlen
             got_it = NC_GET_ATTRIB( dset, ivcd, 'parent_grid',
     .                           do_warn, vname(:vlen), maxlen, attlen, 
     .                           attoutflag, buff, val)
             IF ( got_it ) THEN
                got_it = CD_GET_BOUNDS(cdfid, ivcd, 
     .                     ds_grid_start(1,ivtm), ds_grid_end(1,ivtm), 
     .                     recdim, status )
                IF ( status .NE. merr_ok ) GOTO 5900
             ENDIF

          ENDIF ! nvatts > 0

* officially claim this variable slot
* 7/98 - save in case-sensitive form at this point -- will check over later
*        ds_var_code(ivtm) = vname
         CALL string_array_modify(ds_var_code_head, ivtm, 
     .                            vname, LEN(vname))

* successful completion - variable cataloged
 1000	status = merr_ok
	RETURN

* no variable created
 200	ivtm = 0	! flag for no success
	GOTO 1000

* ERROR MESSAGES
 5100	CALL TM_ERRMSG ( merr_varlim, status, 'CD_GET_1_DEP_VAR',
     .			 dset, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxvars)),
     .			 no_errstring, *5900)

 5200   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_GET_1_DEP_VAR', cdfid,
     .        no_varid, no_errstring, no_errstring, *5900 )

* error exit
 5900	ivtm = 0
	RETURN
	END
