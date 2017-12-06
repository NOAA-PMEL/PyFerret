

      SUBROUTINE ADD_ATTRIBUTE( grid_data, mr, varattname, dset, 
     .         attype_spec, attoutflag, new_att, status )

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
*  add a new attribute to a variable

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP
* 9/2005
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* V698  5/16 *acm* Ticket 2416 allow all output types for DEFINE ATT.
* v73+ *acm* 12/2017 Issue # 1846, disallow defining new scale_factor, add_offset
*                    on file variables

      include 'tmap_dims.parm'
      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'netcdf.inc'

* local parameter declarations:
      INTEGER  max_line_len
      PARAMETER ( max_line_len = 2048 )

* calling argument declarations:

      LOGICAL new_att
      INTEGER mr, attype_spec, dset, attoutflag, status
      REAL grid_data(mr_lo_ss(mr,1):mr_hi_ss(mr,1),
     .               mr_lo_ss(mr,2):mr_hi_ss(mr,2),
     .               mr_lo_ss(mr,3):mr_hi_ss(mr,3),
     .               mr_lo_ss(mr,4):mr_hi_ss(mr,4),
     .               mr_lo_ss(mr,5):mr_hi_ss(mr,5),
     .               mr_lo_ss(mr,6):mr_hi_ss(mr,6))
      CHARACTER*(*) varattname

* local variable declarations: 
      
      LOGICAL valid, do_err
      INTEGER MR_DIM_LEN, TM_LENSTR1, STR_SAME, varid, idim, ndims, attlen,
     .        attype, span, lo(nferdims), hi(nferdims), del(nferdims), slen, vlen
      REAL dummy
      CHARACTER GET_STRING_ELEMENT*2048, TM_FMT*2048, attbuff*2048, 
     .          varname*512, attname*128

      vlen = TM_LENSTR1(varattname)  
      new_att = .TRUE.

! Check that the expresssion is 1-D and if it is a string, just 1 value.

      ndims = 0
      attlen = 0
      DO 30 idim = 1, nferdims
         span = MR_DIM_LEN(idim, mr )
         valid = mr_lo_ss( mr,idim ) .NE. unspecified_int4
         IF (span .GT. 0 .AND. valid ) ndims = ndims + 1
         attlen = MAX(attlen, span)
 30   CONTINUE

* * * * * DETERMINE INDEX LIMITS * * *
      DO 40 idim = 1, nferdims
         lo(idim) = mr_lo_ss( mr, idim )
         hi(idim) = mr_hi_ss( mr, idim )
         del(idim)= 1
 40   CONTINUE

      IF (ndims .LT. 0 .OR. ndims .GT. 1) GOTO 5100

* Reconcile data types for the attribute. If requested to add a float attribute
* but the expression is a string, it is an error.  If a string was requested,
* we can convert a numeric value to string below.

      IF (attype_spec .EQ. ptype_unknown) THEN
         attype = NCFLOAT
         IF (mr_type(mr) .EQ. ptype_string) attype = NCCHAR 
      ELSE
         attype = attype_spec
      ENDIF
      IF (attype .NE. NCCHAR .AND. 
     .    mr_type(mr) .EQ. ptype_string) GOTO 5300

* See if the attribute already exists for the variable. (If it does
* then one can use SET ATT). The flag do_err is false so no error message 
* is issued in break_varattname if attr does not exist.

      do_err = .FALSE.
      CALL BREAK_VARATTNAME ( varattname, dset, varname, 
     .              attname, varid, do_err, status )

* Attributes for packing data may be defined or redfined on user-variables only.
* Otherwise, if on a file variable, it's ambiguous: should they be applied on reading? 
* or only on writing?

      IF (dset .NE. pdset_uvars) THEN
        IF ( STR_SAME(attname, 'scale_factor').EQ.0  .OR. 
     .       STR_SAME(attname, 'add_offset').EQ.0) GOTO 5500
      ENDIF

      IF (status .NE. ferr_ok) THEN
         IF (attype .EQ. NCCHAR ) THEN

* Requested string but provided a numeric expression.
* Use first value of expression.

            IF (mr_type(mr) .NE. ptype_string) THEN
               attbuff = TM_FMT( 
     .             grid_data(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),
     .             8, 40, slen)
               IF (attlen .GT. 1) THEN 
                  CALL WARN ('Requested string attribute using a ',
     .            'numeric expression. Using first value and ',
     .            'converting to string')
                  attlen = 1
               ENDIF

            ELSE
               IF (attlen .GT. 1) GOTO 5200
               attbuff = GET_STRING_ELEMENT(
     .           lo(1), hi(1), lo(2), hi(2), lo(3), hi(3), 
     .	         lo(4), hi(4), lo(5), hi(5), lo(6), hi(6),
     .           grid_data, lo(1), lo(2), lo(3), lo(4), lo(5), lo(6),
     .	         max_line_len, slen)

               dummy = 1.
               attlen = TM_LENSTR1(attbuff)    

            ENDIF
            CALL CD_PUT_NEW_ATTR (dset, varid, attname, attype, attlen, 
     .         attoutflag, attbuff, dummy, status)  

         ELSE 
            CALL CD_PUT_NEW_ATTR (dset, varid, attname, attype, attlen, 
     .         attoutflag, attbuff, 
     .         grid_data(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)), status)      
         ENDIF

!  If attribute already exists, error

      ELSE
c         GOTO 5400
         new_att = .FALSE.
      ENDIF

 5000 RETURN
 5100 attbuff = varattname(:vlen)
      CALL ERRMSG( ferr_syntax, status,
     .  'attribute values must be 1 dimensional: '//
     .   attbuff(:vlen),*5000 )

 5200 attbuff = varattname(:vlen)
      CALL ERRMSG( ferr_syntax, status,
     .  'String attribute may contain just one string: '//
     .  attbuff(:vlen),*5000 )

 5300 attbuff = varattname(:vlen)
      CALL ERRMSG( ferr_syntax, status,
     .  'Attribute type does not match expression: '//
     .   attbuff(:vlen),*5000)

 5400 attbuff = varattname(:vlen)
      CALL ERRMSG( ferr_syntax, status,
     .  'Attribute already exists: '//attbuff(:vlen)//
     .  ' Use SET ATTRIBUTE to change an attribute',*5000)

 5500 attbuff = varattname(:vlen)
      CALL ERRMSG( ferr_syntax, status,
     .  'Defining scale_factor or add_offset attributes not allowed on file variables: '//attbuff(:vlen)
     .  ,*5000)

      END
