	SUBROUTINE CD_GET_TMAP_PARMS ( dset, cdfid, status )

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
* read special TMAP global attributes from the open netCDF file
* these attributes are:
* TMAP_expnum = "0056"
* TMAP_slice = "  SA"
* TMAP_source = "PHIL/SEIG"
* TMAP_subsource = "TROPICAL PACIFIC"
* TMAP_**anything_else**  = auxiliary parameter

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* V600 *ACM* attribute control. 
*             Replace NCINQ with NCF_INQ_DS
*             Change call to CD_GET_ATTRIB to NC_GET_ATTRIB - get attrib 
*             from linked list structure

* argument definitions:
*       dset    - pointer to TMAP data set
*       cdfid   - returned pointer to open CDF file
*	status	- return status

* argument declarations:
	INTEGER	      dset, cdfid, dset_num, status

* include files

        include 'netcdf.decl'
#include "netcdf.inc"
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xio.cmn_text'
        external xio_data

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      LOGICAL NC_GET_ATTRIB, got_it, do_warn
      PARAMETER (do_warn = .TRUE.)

      INTEGER NCF_INQ_DS,
     .        nparm, iatt,  ndims, nvars, ngatts, recdim, 
     .        i, varid, maxlen, attlen, attoutflag
      CHARACTER*13 TM_STRING
      CHARACTER    name*64, tmap_att*64, buff*2048
      REAL vals(10)

      INTEGER tt                ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* initialize
        nparm = 0
        DO 10 i = 1, num_parm
           ds_parm_name(i, dset) = ' '
           ds_parm_text(i, dset) = ' '
 10     CONTINUE
	ds_source_class(dset)	= ' '
	ds_source(dset)		= ' '
	ds_subsource(dset)	= ' '
	ds_expnum(dset)		= ' '
	ds_modnum(dset)		= ' '

* get number of global attributes from stored linked list structure

        dset_num = dset
        IF (dset_num .LT. -1) dset_num = -1
        status = NCF_INQ_DS (dset_num, ndims, nvars, ngatts, recdim )

* get global attribute pseudo-variable id

        IF (ngatts .GT. 0) THEN
           CALL CD_GET_VAR_ID (dset, '.', varid, status)

* loop through global attributes looking for ones beginning with TMAP
           DO 100 iatt = 1, ngatts

* get global attribute name
              name = ' '
              CALL CD_GET_VAR_ID (dset, '.', varid, status )
              CALL CD_GET_VAR_ATT_NAME( dset, varid, iatt, name, status)
              IF ( name(:5) .NE. 'TMAP_' ) GOTO 100

* get its value
              maxlen = 2048
              got_it = NC_GET_ATTRIB ( dset, varid, name, do_warn, 
     .                      '.', maxlen, attlen, attoutflag, 
     .                      buff, vals)

* what is it ?
              tmap_att = name(6:)
              IF ( tmap_att(:6) .EQ. 'expnum' ) THEN
                 ds_expnum(dset) = buff
              ELSEIF ( tmap_att(:5) .EQ. 'slice' ) THEN    ! "modnum" in GT sets
                 ds_modnum(dset) = buff
              ELSEIF ( tmap_att(:6) .EQ. 'source' ) THEN
                 ds_source(dset) = buff
              ELSEIF ( tmap_att(:9) .EQ. 'subsource' ) THEN
                 ds_subsource(dset) = buff
              ELSE
                 nparm = nparm + 1
                 IF ( nparm .GT. num_parm ) GOTO 5100
                 ds_parm_name(nparm, dset) = tmap_att
                 ds_parm_text(nparm, dset) = buff
              ENDIF

 100       CONTINUE
        ENDIF

* successful completion
 1000   status = merr_ok
        RETURN

* error exit
 5100      CALL TM_NOTE( 'maximum number of TMAP CDF attributes '
     .          //'exceeded: '//TM_STRING(DBLE(num_parm)), tt )
           GOTO 1000   ! not fatal
 5200      CALL TM_NOTE( 'dataset not found', tt )
           RETURN

           END
