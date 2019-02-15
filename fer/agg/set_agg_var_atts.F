    	SUBROUTINE SET_AGG_VAR_ATTS(agg_dset, vname, aggv_slot, nsets,
     .                              ref_fvar, memb_set, status)

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
* assign attributes to an aggregated variable if the same, identical attributes
* are found on all member variables
* (units, title, and bad_flag are taken care of elsewhere)


* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V698 *sh* 2/16 
* V698 *acm* 2/16 LET/D variables are now stored with their associated datset
*                 in the List structure so can get their attribute info
* V720 *acm* 5/17 Implement specific calls in place of CD_GET_VAR_INFO

        IMPLICIT NONE
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'ferret.parm'
        INCLUDE 'errmsg.parm'      
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'xvariables.cmn'
        INCLUDE 'xcontext.cmn'
        INCLUDE 'xdset_info.cmn_text'
        INCLUDE 'netcdf.inc'

* calling arguments
*  agg_dset     - dset no of the aggregation (input)
*  vname        - name of the aggregated variable (input)
*  aggv_slot    - index of agg var in ds_ common (input)
*  nsets        - number of member datasets in aggregation (input)
*  ref_fvar     - reference file variable serving as the pattern (input)
*  memb_set     - list of member dataset numbers (input)
*  status       - return status (output)

* calling argument declarations
    	INTEGER agg_dset, aggv_slot, nsets, 
     .          ref_fvar, memb_set(*), status
        CHARACTER*128 vname

* Local variable declarations
        INTEGER maxlen
        PARAMETER (maxlen = 2048)

        LOGICAL NC_GET_ATTRIB, got_it
        INTEGER TM_LENSTR1, vlen, ref_dset, ref_id, var_id,
     .          nvatts, att, imemb, attype, attlen, attoutflag,
     .          m_attid, m_attype, m_attlen, m_attoutflag,
     .          m_dset, m_cat, m_grid, m_var, m_ens_ax,
     .          all_outflag, memb_id(nsets), i, iatt
        REAL    attvals(50), m_attvals(50)
        CHARACTER attname*128
        CHARACTER*(maxlen) attstr, m_attstr


* initialize
        vlen = TM_LENSTR1(vname)
        ref_dset = ds_var_setnum(ref_fvar)

* Get var id for reference variable
        CALL CD_GET_VAR_ID (ref_dset, vname(:vlen), ref_id, status)
        IF (status .NE. ferr_ok) GOTO 5200

* Get var id for aggregated variable
        CALL CD_GET_VAR_ID (agg_dset, vname(:vlen), var_id, status)
        IF (status .NE. ferr_ok) GOTO 5200

* Get var ids for member variables
        imemb = 1
        DO imemb = 1, nsets

* skip dummy datasets
           IF (memb_set(imemb) .EQ. pdset_dummy) THEN
              memb_id(imemb) = unspecified_int4
              CYCLE
           ENDIF

* LET/D vars are represented in the List structure of their assigned
* dataset. Treat as for any file variable
	   CALL CD_GET_AGG_VAR_INFO (agg_dset, vname(:vlen), imemb, 
     .         m_cat, m_dset, m_grid, m_ens_ax, m_var, status)
           IF (status .NE. ferr_ok) GOTO 5200
           CALL CD_GET_VAR_ID (memb_set(imemb), vname(:vlen),
     .                         memb_id(imemb), status)
           IF (status .NE. ferr_ok) GOTO 5200

        ENDDO

* Get number of attributes from the reference variable
        CALL CD_GET_VAR_NATTS (ref_dset, ref_id, vname(:vlen),
     .                         nvatts, status)

**** loop through the attribute list from the reference variable
        attloop: DO iatt = 1, nvatts
           CALL CD_GET_VAR_ATT_NAME (ref_dset, ref_id, iatt, attname, status)

* Ignore basic attributes. These are already defined.
*?? Check the attribute values from each member?
           IF (attname .EQ. 'units'
     .    .OR. attname .EQ. 'long_name'
     .    .OR. attname .EQ. '_FillValue'
     .    .OR. attname .EQ. 'missing_value') CYCLE

* get the reference attribute info
           CALL CD_GET_VAR_ATT_INFO (ref_dset, ref_id, iatt, attname, 
     .               attype, attlen, attoutflag, status )
           IF (status .NE. ferr_ok) GOTO 5200

* get the reference attribute value and info
           got_it = NC_GET_ATTRIB ( ref_dset, ref_id, attname,
     .                              .TRUE., vname(:vlen), maxlen,
     .                              attlen, attoutflag,
     .                              attstr, attvals )
           IF (.NOT.got_it) GOTO 5200
	   
* ensure that the attribute is identical in all member variables

           DO imemb = 1, nsets

* ... skip over dummy members
              IF (memb_set(imemb) .EQ. pdset_dummy) CYCLE

! ... skip over user variables
              IF (memb_id(imemb)  .EQ. unspecified_int4) CYCLE

* ... get the attribute id in the member
              CALL CD_GET_VAR_ATT_ID (memb_set(imemb), memb_id(imemb),
     .                                attname, m_attid, status)
              IF (status.EQ.atom_not_found) CYCLE attloop

* ... get the attribute info for the member
              CALL CD_GET_VAR_ATT_INFO (memb_set(imemb), memb_id(imemb),
     .                                  m_attid, attname, m_attype, 
     .                                  m_attlen, m_attoutflag, status )
              IF (status .NE. ferr_ok) GOTO 5200
              IF (m_attype .NE. attype
     .       .OR. m_attlen .NE. attlen ) CYCLE attloop

* get the attribute value for the member
              got_it = NC_GET_ATTRIB (memb_set(imemb), memb_id(imemb),
     .                              attname,
     .                              .TRUE., vname(:vlen), maxlen,
     .                              m_attlen, m_attoutflag,
     .                              m_attstr, m_attvals )
              IF (.NOT.got_it) GOTO 5200

* check that the value matches the reference
              IF (attype .EQ. NCCHAR) THEN
* ... check that string values are identical
                 IF (m_attstr .NE. attstr)  CYCLE attloop
              ELSE
* ... check that numeric values are identical
                 DO i = 1, attlen
                    IF (attvals(i) .NE. m_attvals(i)) CYCLE attloop
                 ENDDO
              ENDIF

           ENDDO   ! done cmparing with member attributes

* transfer this attribute from the reference variable to the aggregate variable
	   CALL CD_PUT_NEW_ATTR (agg_dset, var_id, attname, attype, 
     .                  attlen, attoutflag, attstr, attvals, status )

        ENDDO attloop

* Do not call UPDATE_ATTRIBUTES. The title, units, missing etc were
* already set in defining the variable in the aggregation.
cc        CALL UPDATE_ATTRIBUTES (agg_dset, vname, aggv_slot, status)
cc        IF (status .NE. ferr_ok) GOTO 5200

* success
* (status may have been set to atom_not_found on a missing attribute.
*  Set to ferr_ok.)

	status = ferr_ok
        RETURN

* error exits
 5000   RETURN

! 5100   CALL ERRMSG (ferr_var_not_in_set, status, vname(:vlen),*5000)
!
 5200   CALL ERRMSG (ferr_internal, status,
     .               'attListCrptn: '//vname(:vlen),*5000)
	END
