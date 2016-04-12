	CHARACTER*(*) FUNCTION CX_DIM_STR( idim,cx,to_word,full_prec,slen )

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
* create a string describing the region and transformation along an axis
* for example: X=160E:140W@SBX:5
* the argument full_prec ("full precision") indicates number of decimals for
* world coordinates (see LOHI_WORLD) or skip/not skip leading integer 0's

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 9/26/87
* V200:  5/19/89 - re-ordered _ss args
*		 - changed LEFINT from *4 to *8
*		 - allowed leading zeros as needed by size of cx_hi_ss
* V230:  6/22/92 - if ss limits are unspec. then use world limits regardless
*                  of cx_by_ss (needed for CDF_LIST documentation)
*        8/18/92 - eliminated string(1:) notation for MAC compatibility
* V533:  6/01 - allow for the possibility of negative subscripts
* V600:  1/06 *acm*  prevent values of LOG10(0.0) 
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V698 *sh* 3/2016 document a true E ensemble axis with dataset identifier

	include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'xcontext.cmn'
	include 'xtext_info.cmn'
	include 'xdset_info.cmn_text'

* calling argument declarations
	LOGICAL	full_prec
	INTEGER	idim, cx, slen
	CHARACTER*(*) to_word

* local variable declarations:
	LOGICAL TM_HAS_STRING, its_ensemble
	CHARACTER LOHI_WORLD*44
	CHARACTER ALG_TRANS_CODE*3, ARG_STRING*12, LEFINT*8, buff*8
	INTEGER TM_LENSTR1, ENSEMBLE_MEMBER_DSET
	INTEGER llen, ndigits, cx_dim_str_len, dset, slen2, memb_dset

* initialize - put in "=" and blank out string
        cx_dim_str_len = LEN( CX_DIM_STR )
	CX_DIM_STR = '*='

* initialize special case of documenting an ensemble axis
	its_ensemble = .FALSE.
	IF (cx .GT. cx_buff) THEN
	   dset = cx_data_set(cx)
	   its_ensemble = idim .EQ. E_dim
     .		    .AND. dset.GT.pdset_irrelevant
     .	            .AND. dset.LE.max_gfdl_dsets
 	   IF (its_ensemble) THEN
	      its_ensemble = TM_HAS_STRING(ds_type(dset), 'ENS')
	   ENDIF
	ENDIF

* create "lo_limit" or "lo:hi" in subscript or world coords

* SUBSCRIPT NOTATION (I,J,K or L)
* ( added ".AND. cx_lo_ss(cx,idim) ..." 6/22 *sh*)
	IF ( cx_by_ss( idim, cx )
     . .AND. cx_lo_ss(cx,idim) .NE. unspecified_int4 ) THEN
	   CX_DIM_STR(1:1) = ss_dim_name( idim ) 
* lo limit
	   IF ( full_prec ) THEN
* ... only the minimum number of digits needed - maximum readability
	      CX_DIM_STR(3:cx_dim_str_len) =
     .                               LEFINT( cx_lo_ss(cx,idim), slen )
	      slen = slen + 2
	   ELSE
* ... uses leading zeros to ensure filenames generated automatically will colate
	      CX_DIM_STR(3:cx_dim_str_len) = '0000000000000000'
              IF (REAL(ABS(cx_hi_ss(cx,idim))) .NE. 0.) THEN
	         ndigits = LOG10( REAL(ABS(cx_hi_ss(cx,idim))) ) + 1.00001
              ELSE
	         ndigits = 1.00001
              ENDIF
	      IF ( cx_hi_ss(cx,idim) .LT. 0 ) ndigits = ndigits + 1
	      slen = 2 + ndigits	! right-most character
	      buff = LEFINT( cx_lo_ss(cx,idim), llen )
	      CX_DIM_STR(slen-llen+1:slen)=buff(:llen)
	   ENDIF
* hi limit - similar logic
	   IF ( cx_hi_ss(cx,idim) .NE. cx_lo_ss(cx,idim) ) THEN
	      slen = slen + 1
	      CX_DIM_STR(slen:slen) = ':'	
	      IF ( full_prec ) THEN
	         CX_DIM_STR(slen+1:cx_dim_str_len) =
     .                LEFINT(cx_hi_ss(cx,idim), llen)
	         slen = slen + 1 + llen
	      ELSE
	         buff = LEFINT( cx_hi_ss(cx,idim), llen )
	         slen = slen + ndigits
	         CX_DIM_STR(slen-llen+1:slen)=buff(:llen)
	      ENDIF
	   ENDIF
	   IF ( .NOT.full_prec ) CX_DIM_STR(slen+1:cx_dim_str_len)= ' '

* WORLD COORDINATE NOTATION
	ELSE
	   CX_DIM_STR(1:1) = ww_dim_name( idim )
	   CX_DIM_STR(3:cx_dim_str_len)  =
     .                 LOHI_WORLD( idim, cx, ':', full_prec, slen )
	   slen = slen + 2
	ENDIF

* transformation, if any ... "@ttt:n"
	IF ( cx_trans( idim, cx ) .NE. trans_no_transform ) THEN
	   CX_DIM_STR = CX_DIM_STR(1:slen) // '@' //
     .				ALG_TRANS_CODE( cx_trans( idim, cx )) //
     .				ARG_STRING( cx_trans_arg( idim, cx ))
	   slen = TM_LENSTR1( CX_DIM_STR )
	ENDIF

* a single ensemble member name
	IF ( its_ensemble
     .  .AND.cx_lo_ss(cx,idim).NE.unspecified_int4
     .  .AND.cx_lo_ss(cx,idim).EQ.cx_lo_ss(cx,idim) ) THEN


* ... get the special string that indicates the ensemble member
	   memb_dset = ENSEMBLE_MEMBER_DSET(dset, cx_lo_ss(cx,idim))
	   IF (  memb_dset.GT.pdset_irrelevant
     .	    .AND.memb_dset.LE.max_gfdl_dsets) THEN
	      slen2 = TM_LENSTR1( ds_name(memb_dset) )
	      slen2 = MIN(slen2, 12)  ! no more than 12 chars
	      CX_DIM_STR = CX_DIM_STR(1:slen) // ' ('
     .			// ds_name(memb_dset)(:slen2) // ')'	
	      slen = TM_LENSTR1( CX_DIM_STR )
	   ENDIF
	ENDIF
	
	RETURN
	END
