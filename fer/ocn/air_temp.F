      SUBROUTINE AIR_TEMP ( sst, msst, cair, mcair, air, mair )

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
* compute air temperature: either climatological field (unmodified)
*			   or SST-const south of j=91 and climatol. to the north

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/12/88
* revision 0.1 - 7/2/88  - allow multiple air temp climatologies
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'tmap_format/tmap_dims.parm'
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xdset_parms.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XDSET_PARMS.CMN'
#endif

* calling argument declarations:
	INTEGER	mair, msst, mcair
	REAL     sst( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		cair( m2lox:m2hix,m2loy:m2hiy,            m2lot:m2hit ),
     .		 air( m3lox:m3hix,m3loy:m3hiy,            m3lot:m3hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, l, dset
	REAL	bad_cair, bad_air, bad_sst, dairt

* --- end of introductory code ---
* initialize
	dset    = mr_data_set( mair )

* limits for calculation
	i_lo = mr_lo_s1(mair)
	i_hi = mr_hi_s1(mair)
	j_lo = mr_lo_s2(mair)
	j_hi = mr_hi_s2(mair)

* flag for bad/missing data
	bad_cair  = mr_bad_data( mcair )
	bad_air   = mr_bad_data( mair )
	bad_sst   = mr_bad_data( msst )

* constant air/sst temperature difference
	dairt = dp_delta_air_sst( dset )


	IF ( .NOT.dp_const_air_sst(dset) ) THEN

* climatological air temp
	   DO 100 l = mr_lo_s4(mair), mr_hi_s4(mair)
	   DO 100 j = j_lo, j_hi
	   DO 100 i = i_lo, i_hi

	      IF ( cair(i,j,l) .EQ. bad_cair ) THEN
	         air(i,j,l) = bad_air
	      ELSE
	         air(i,j,l) = cair(i,j,l)
	      ENDIF

 100	   CONTINUE

	ELSE
* constant air-sst difference in south, climatology in north
	   DO 200 l = mr_lo_s4(mair), mr_hi_s4(mair)
	   DO 200 j = j_lo, j_hi
	   DO 200 i = i_lo, i_hi

	      IF ( j .LT. 91 ) THEN
	         IF ( sst(i,j,1,l) .EQ. bad_sst ) THEN
	            air(i,j,l) = bad_air
	         ELSE
	            air(i,j,l) = sst(i,j,1,l) - dairt
	         ENDIF
	      ELSE
	         IF ( cair(i,j,l) .EQ. bad_cair ) THEN
	            air(i,j,l) = bad_air
	         ELSE
	            air(i,j,l) = cair(i,j,l)
	         ENDIF
	      ENDIF

 200	   CONTINUE

	ENDIF
	RETURN
	END
