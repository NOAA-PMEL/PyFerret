	SUBROUTINE SHOW_DATA_SET_PARAMS( lun, dset )

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
* describe the diagnostic parameters of the given data set

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -  1/10/89 - based on SHOW_DATA_SET 0.2
* Unix/RISC port 3/91 *sh* - cannot use function in WRITE list
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI

#ifdef unix
	include 'tmap_dims.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret.parm'
	include	'xdset_parms.cmn'
	include	'xrisc.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XDSET_PARMS.CMN'
#endif

* calling argument declarations:
	INTEGER	lun, dset
	
* local parameters:
	INTEGER	sigdig, mxlen
	PARAMETER     ( sigdig = 5, mxlen = 20 )

* local variable declarations:
	INTEGER 	TM_LENSTR1, dlen, cairt
	CHARACTER*4	buffer*24, TM_FMT*20

* data set title (and possible modifier)
	CALL SPLIT_LIST(pttmode_explct, lun,
     .		'     '//ds_title(dset)(:TM_LENSTR1(ds_title(dset)))
     .								,0)
	IF ( ds_mod_title(dset) .NE. ' ' ) CALL SPLIT_LIST
     .		(pttmode_explct, lun,
     .	'     '//ds_mod_title(dset)(:TM_LENSTR1(ds_mod_title(dset)))
     .								,0)

* describe the "type" of data set
	IF ( dp_phil_sieg(dset) .OR. dp_cox(dset) ) CALL SPLIT_LIST
     .	(pttmode_explct, lun,
     .	'             Philander/Seigel diagnostic variables available'
     .					,0)

* special Philander/Seigel parameters:
	IF ( dp_phil_sieg(dset) ) THEN
* ... min. wind speed
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		'           minimum wind speed: '//
     .		TM_FMT(dp_min_wind(dset),sigdig,mxlen,dlen), 0 )
* ... heat flux parameterization
	   risc_buff = ' '
	   IF ( dp_phil_qflux(dset) ) THEN
	      WRITE (risc_buff, 3110) 'from Philander/Seigel'
	   ELSEIF ( dp_ramp1_qflux(dset) ) THEN
	      WRITE (risc_buff, 3110) 'single ramp - attenuate above 27 deg.'
	   ELSEIF ( dp_ramp2_qflux(dset) ) THEN
	      WRITE (risc_buff, 3110) 'double ramp - at T>27 and T<26'
	   ENDIF
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0 )
* ... constant air/sst temp. delta
	   IF ( dp_const_air_sst(dset) ) THEN
	      CALL SPLIT_LIST(pttmode_explct, lun,
     .			'           constant air/SST difference: '//
     .			TM_FMT(dp_delta_air_sst(dset),sigdig,mxlen,dlen), 0)
           ENDIF
* ... climatological air temp field
	   cairt = dp_clim_air_sst( dset )		! cairt needed for calcs
	   risc_buff = ' '
	   WRITE (risc_buff,3130) dp_airt_name( cairt )
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0 )
* ... Am factor
	   buffer = TM_FMT( dp_Am_factor(dset), sigdig, mxlen, dlen )
	   risc_buff = ' '
	   WRITE (risc_buff,3140) buffer 
	   CALL SPLIT_LIST(pttmode_explct, lun, risc_buff, 0 )
	ENDIF
 3110	FORMAT ( T12,'surface radiation parameterization: ',A)
 3120	FORMAT ( T12,'constant air/SST difference: ',A)
 3130	FORMAT ( T12,A,' climatological air temperature field')
 3140	FORMAT ( T12,'horizontal momentum Am = ',A )

	RETURN
	END
