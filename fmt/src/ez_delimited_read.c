/*
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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL
,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.  
*
*/

/* Revision history
   V530 - *sh* added ability to read dates formatted as YYYYMMDD
   V540 *sh* 10/01 - when analyzing a file insist on ENUF_IN_A_ROW records
                     successfully evaluated
   v542 *kob* 10/02 - fix up FTYP_DATE and FTYP_EURODATE so that it can
                      handle four digit years.  Also allow date in euro
		      format yyyyddmm to be acceptable
   v600 *acm* 4/06   change call to days_from_day0 in DecodeRec becaues of
                     problems with 64-bit build
 *acm*  1/12      - Ferret 6.8 ifdef double_p for double-precision ferret, see the
					 definition of macro DFTYPE in ferretmacros.h.
 *acm*  5/13      - ticket 2066: For double-precision Ferret reading numeric data,
                    sscanf needs %lf instead of %f, and dummy needs to be double prec.
  V701 7/16 *acm*   ticket 2450: add date-time and euro-date-time field types
                    ticket 2448: for 2-digit years, put years prior to 50 into the 2000s
                    ticket 2449: report incorrect choice of date/ eurodate as an error
  V702 10/16 *acm*  ticket 2472: Allow yyyy/dd/mm in any of the date types.   

*/



/*
  Code to perform tab, comma, etc delimited reads from Ferret
  Top level Routines are
  anal_file_ - determines the number of fields and types of each field
  decode_file_ - reads an entire file based upon analysis suppied

  Support routines are
  analRec  - performs analysis of a single record
            ... possibly with missing fields
  decodeRec - reads a single record of the input file
  nexstrtok - breaks input into successive fields
 */


/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */

#include <wchar.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "ferretmacros.h"
#include "ez_delimited_read.h"


/*
 *
 decode_file - skip header & read an entire file based upon analysis suppied
 decode_file_ arguments
 fname - input filename (null terminated)
 recptr - buffer to hold file records
 reclen - size of record buffer
 delims - string containing delimiter in file (note that blank is
          considered a special delimiter representing any number of blanks)
 skip - number of heading record in file to be skipped
 maxrec - maximum number of records to read (beyond skipped header)
 nrec - out number of records actually read
 nfields - number of fields to decode on each record
 field_types - type of each field (input)
 mrlist - list of integers pointing to Ferret memory blocks
 memptr - pointer to base of Ferret "heap" storage
 mr_blk1 - memory chunk numbers indexed by mr_list
 mblk_size - chunk size within Ferret heap
 mr_bad_flags - missing value flags indexed by mr_list
 *
 */


void FORTRAN(decode_file_jacket)
		( char* fname, char *recptr, char *delims, int *skip,
		  int* maxrec, int* reclen, int* nfields,
		  int field_type[], int* nrec,
		  int mrlist[], DFTYPE *memptr, int mr_blk1[], int* mblk_size,
		  DFTYPE mr_bad_flags[], char ***mr_c_ptr, int* status)

{

  DFTYPE **numeric_fields  = (DFTYPE **) malloc(sizeof(DFTYPE*) * (*nfields));
  DFTYPE *bad_flags        = (DFTYPE *)  malloc(sizeof(DFTYPE) * (*nfields));

  char ***text_fields     = (char ***) malloc(sizeof(char**) * (*nfields));
  int i, mr;
  int pinc = 8/sizeof(char*);  /* pointers spaced 8 bytes apart */

  for (i=0; i<(*nfields); i++)
    {
      mr = mrlist[i] - 1;  /* -1 for C indexing */
      /* 
	 compute separate pointer arrays for numeric and text fields
      */

      numeric_fields[i] = (DFTYPE *) NULL;
      text_fields[i] = (char**) NULL;
      
      if (field_type[i] == FTYP_CHARACTER )
	{
	  /* *kob* make sure were using size of real*4 float */
	  text_fields[i] = (char**) (memptr + ((mr_blk1[mr]-1)*(*mblk_size)*4)/sizeof(float));
	  mr_c_ptr[mr*pinc] = text_fields[i];
	}
      else if (field_type[i] != FTYP_MISSING )
	{
	  numeric_fields[i] = memptr + (mr_blk1[mr]-1)*(*mblk_size);
	  mr_c_ptr[mr*pinc] = (char**) NULL;
	}
      /*
	isolate the bad data flags that correspond to the numeric fields
      */
      if ( (field_type[i]!=FTYP_MISSING) && (field_type[i]!=FTYP_CHARACTER) )
	bad_flags[i] = mr_bad_flags[mr];
      else
	bad_flags[i] = 0.0;
    }

  /*
    at last we actually read the file
  */
  decode_file (fname, recptr, delims, skip, 
	       maxrec, reclen, nfields,
	       field_type, nrec,
	       numeric_fields, text_fields, bad_flags, status);

  free(numeric_fields);
  free(text_fields);
  free(bad_flags);

  return;
}


/*
 *
 decode_file - skip header & read an entire file based upon analysis suppied
 decode_file_ arguments
 fname - input filena0me
 recptr - buffer to hold file records
 reclen - size of record buffer
 delims - string containing delimiter in file (note that blank is
          considered a special delimiter representing any number of blanks)
 skip - number of heading record in file to be skipped
 maxrec - maximum number of records to read (beyond skipped header)
 nrec - out number of records actually read
 nfields - number of fields to decode on each record
 field_types - type of each field (input)
 numeric_fields - array to be filled with numeric field values
                (2D - # of vareiables by # of records to read)
 bad_flags - missing value flags (used for numeric fields, only) 
 text_fields - array to be filled with string pointers
 (2D - # of vareiables by # of records to read)
 *
 */


int decode_file (char* fname, char *recptr, char *delims, int *skip, 
			  int* maxrec, int* reclen, int* nfields,
			  int field_type[], int* nrec, DFTYPE** numeric_fields,
			  char*** text_fields, DFTYPE bad_flags[], int* status)

{

  FILE *fp;
  int slen, i;
  int pinc = 8/sizeof(char*);  /* pointers spaced 8 bytes apart */

  *nrec = 0;

  fp = fopen(fname,"r");

  /* skip initial records */
  for (i=0; i<*skip; i++)
    {  
      if ( fgets(recptr,*reclen,fp) == NULL )
        break;
    }

  while (!feof(fp) && (*nrec)<(*maxrec))
    {

      if ( fgets(recptr,*reclen,fp) )
	{
	  /* skip leading blanks */
	  while (*recptr==' ')
	    recptr++;

	  /* overwrite the newline record terminator with a NULL */
	  if ((slen = strlen(recptr)) > 0)
	    if (recptr[slen-1] == '\n')
	      recptr[slen-1] = '\0';
	  
	  decodeRec(recptr, delims, nfields, field_type, *nrec,
		    numeric_fields, text_fields, bad_flags, status);

#ifdef diagnostic_output	  /* ************* */
	  for (i=0; i<(*nfields); i++)
	    if (field_type[i] == FTYP_CHARACTER )
	      printf( "%d character: %s\n",i, (*(text_fields+i))[(*nrec)*pinc] );
	    else if (field_type[i] != FTYP_MISSING)
	      printf( "%d numeric: %f\n",i,(*(numeric_fields+i))[(*nrec)] );
#endif                            /* ************* */
	   if (*status != 3) return 0;
	   
	  (*nrec)++;
	}
    }

  fclose(fp);

  return 0; /* always OK ? */

}
/*
 *
 anal_file_ - determine the number of fields and types of each field
          - if an input record has missing fields (e.g. ",,") then keep looking
	  - if subsequent records produce differing analyses of a field then
	    flag it as a CHARACTER fields
 anal_file_ arguments
 fname - input filena0me
 recptr - buffer to hold file records
 reclen - size of record buffer
 delims - string containing delimiter in file (note that blank is
          considered a special delimiter representing any number of blanks)
 skip - number of heading record in file to be skipped
 maxrec - maximum number of records to read (beyond skipped header)
 nfields - (output) number of fields determined to be in file
 field_types - (output)type of each field
 max_fields - (input) maximum number of fields to consider
 *
 */

int FORTRAN(anal_file) (char* fname, char *recptr, char *delims, int* skip,
			int* maxrec, int* reclen, int* nfields,
			int field_type[], int *max_fields)     
{

#define ENUF_IN_A_ROW 25  /* insist on this many successful record evals */
  FILE *fp;
  int slen, i, rec;
  int nsuccess = 0;

  fp = fopen(fname,"r");

  /* skip initial records */
  for (rec=0; rec<*skip; rec++)
    {  
      if ( fgets(recptr,*reclen,fp) == NULL )
        break;
    }

  /* initially set all field types to missing (no information) */
  for (i=0; i<(*max_fields); i++)
    field_type[i] = FTYP_MISSING;
  *nfields = 0;

  rec = 0;
  while ( !feof(fp) && rec<(*maxrec) )
    {
      if ( fgets(recptr,*reclen,fp) )
	{
	  rec++;

	  /* skip leading blanks */
	  while (*recptr==' ')
	    recptr++;

	  /* overwrite the newline record terminator with a NULL */
	  if ((slen = strlen(recptr)) > 0)
	    recptr[slen-1] = '\0';
	  
	  analRec(recptr, delims, nfields, field_type, *max_fields);

	  /* check for unknown field types */
	  i = 0;
	  while ( i<*nfields && (field_type[i] != FTYP_MISSING) )
	    i++;

	  /* success at analyzing one full record */
	  if (i == *nfields)
	    nsuccess++;
	  else
	    nsuccess = 0;

	  /* success at analyzing 25 records in a row is enough */
	  if (nsuccess > ENUF_IN_A_ROW)
	    {
	      fclose(fp);
	      return 0;
	    }
	}
    }

  /*
    all records in file have been analyzed
  */
  fclose(fp);

  /* only an incomplete analysis of the file was possible
     Note that following 10/01 changes it is no longer certain that the
     analysis contains fields of all missing values. See ENUF_IN_A_ROW
  */
  return FANAL_HAS_MISSING;

}

/*
 *
 decodeRec - parse and return values from a record based upon analysis suppied
 decodeRec arguments
 recptr - buffer holding file record
 delims - string containing delimiter in file (note that blank is
          considered a special delimiter representing any number of blanks)
 nfields - number of fields to decode on each record
 field_types - type of each field (input)
 rec - (input) current record number (points to location in output arrays)
 numeric_fields - array to be filled with numeric field values
                (2D - # of vareiables by # of records to read)
 bad_flags - missing value flags (used for numeric fields, only) 
 text_fields - array to be filled with string pointers
 (2D - # of vareiables by # of records to read)
 *
 */

int decodeRec(char *recptr, char *delims, int* nfields, int field_type[],
	      int rec,
	      DFTYPE** numeric_fields, char*** text_fields, DFTYPE bad_flags[],
	      int* status)
{

  char *p, *pnext, str1[2], errstr[2];
  double dummy;
  DFTYPE rdum;

  int idummy1, idummy2, idummy3, idummy4, idummy5, i;
  char blankstr[] = " ";
  double days_1900 = 59958230400.0 / (60.*60.*24.); 
/*  int days_1900 = 693961;  */

  int pinc = 8/sizeof(char*);  /* pointers spacd 8 bytes apart */
  int slen;     /* kob 12/01 needed to check for numberical string ending in e/E */
  int ndum;
  int break_century;
  double tpart;
  p = recptr;

/* 2-digit years need to be assigned to a century. 
   will break after 2049 or before 1950  (ACM 7/2016 was year 20 */
  break_century = 50; 

  for (i=0; i<*nfields; i++) {
    pnext = nexstrtok(p, delims);
    if ( field_type[i] == FTYP_MISSING ) {
      /* do nothing -- a skipped field */
      ;
    } else if (p==NULL || *p == '\0') {
      /* missing data field */
      if ( field_type[i] == FTYP_CHARACTER ) {
	(*(text_fields+i))[rec*pinc] = (char *) malloc(sizeof(char)*2);
	strcpy( (*(text_fields+i))[rec*pinc], blankstr );
      }
      else {
	(*(numeric_fields+i))[rec] = bad_flags[i];
      }

    } else {
      switch (field_type[i]) {
	
	/* latitude */
      case FTYP_LAT:
	if (sscanf(p,"%lf%1[Nn]%1s",&dummy,str1,errstr) == 2)
	  (*(numeric_fields+i))[rec] = dummy;
	else if (sscanf(p,"%lf%1[Ss]",&dummy,str1) == 2)
	  (*(numeric_fields+i))[rec] = -1 * dummy;
	else if ( sscanf(p,"%lf%1s",&dummy,errstr ) != 1)
	  (*(numeric_fields+i))[rec] = bad_flags[i];
	else if (p[strlen(p)-1] == 'e' || p[strlen(p)-1] == 'E') 
	    (*(numeric_fields+i))[rec] = bad_flags[i];
	else 
	    (*(numeric_fields+i))[rec] = dummy;
	break;
	
	/* longitude */
      case FTYP_LON:
	if (sscanf(p,"%lf%[Ee]%1s",&dummy,str1,errstr) == 2)
	  (*(numeric_fields+i))[rec] = dummy;
	else if (sscanf(p,"%lf%1[Ww]",&dummy,str1) == 2)
	  (*(numeric_fields+i))[rec] = -1 * dummy;
	/* *kob* 12/01 - need to check for a sting ending in e or E and set
	   it to dummy, as above.  osf and linux compiler let such a string through as 
	   a valid number, rather than a longitude */
	else if (p[strlen(p)-1] == 'e' || p[strlen(p)-1] == 'E') 
	    (*(numeric_fields+i))[rec] = dummy;
	else if ( sscanf(p,"%lf%1s",&dummy,errstr ) != 1)
	  (*(numeric_fields+i))[rec] = bad_flags[i];
	else 
	    (*(numeric_fields+i))[rec] = dummy;
	break;
	
	/* date */
      case FTYP_DATE:
	if (sscanf(p,"%d/%d/%d%1s",&idummy1,&idummy2,&idummy3,errstr) == 3) {
	  /* need to check to see if idummy3 which contains the year is in 
	     the form YY or YYYY     *kob*  10/02  */
		 
	  /* Date of 00/00/00 is bad-data */
	  if (idummy1 == 0 || idummy2 == 0 || idummy3 == 0) {
		  (*(numeric_fields+i))[rec] = bad_flags[i];
		  break;
	  }

	  /* check for yyyy/mm/dd */
	  if (idummy1 > 1800) {
		  (*(numeric_fields+i))[rec] =
		  days_from_day0_(&days_1900,&idummy1,&idummy2,&idummy3,&rdum,status);
		  (*(numeric_fields+i))[rec] = rdum;
		  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
		  break;
	  }

	  /* Check for 2-digit or 4-digit year */
	  if (idummy3 < 100) { 
	    if (idummy3 < break_century)   /* assign 2-digit year to century*/
	      idummy3 += 2000;
	    else
	      idummy3 += 1900;
	  }

	  (*(numeric_fields+i))[rec] =
	    days_from_day0_(&days_1900,&idummy3,&idummy1,&idummy2,&rdum,status);
	  (*(numeric_fields+i))[rec] = rdum;
	  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
	/* force dates with dashes "-" to be in yyyy-mm-dd format *kob* */
	} else if (sscanf(p,"%4d-%2d-%2d%1s",
			  &idummy1,&idummy2,&idummy3,errstr) == 3) {
	  (*(numeric_fields+i))[rec] =
	    days_from_day0_(&days_1900,&idummy1,&idummy2,&idummy3,&rdum,status);
	  (*(numeric_fields+i))[rec] = rdum; 
	  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
	  }
	/* yyyyddmm date */
	else if ( (sscanf(p,"%4d%2d%2d%1s",&idummy1,&idummy2,&idummy3,str1)==3)
	      && idummy1>0
	      && idummy2>=1 && idummy2<=12
	      && idummy3>=1 && idummy3<=31 ) {
	  (*(numeric_fields+i))[rec] =
	    days_from_day0_(&days_1900,&idummy1,&idummy2,&idummy3,&rdum,status);
	  (*(numeric_fields+i))[rec] = rdum; }
	else
	  (*(numeric_fields+i))[rec] = bad_flags[i];
	break;

	/* Date and time
	   Parses fields that are date time separated by a space.
	   It would be good to also allow for Z or T separator
	   Currently those can be handled as separate date and time
	   fields with T or Z given as a delimiter. */

	      case FTYP_DATIME:
			  
	if (sscanf(p,"%d/%d/%d %d:%d:%lf%1s",&idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,errstr) >= 4) {

	  /* Date of 00/00/00 in any order is bad-data */
	  if (idummy1 == 0 || idummy2 == 0 || idummy3 == 0) {
		  (*(numeric_fields+i))[rec] = bad_flags[i];
		  break;
	  }

	  ndum = sscanf(p,"%d/%d/%d %d:%d:%lf%1s",&idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,errstr);
	  tpart = -999;
	  if (ndum == 6) tpart = idummy4 + idummy5/60. + dummy/3600.;
	  if (ndum == 5) tpart = idummy4 + idummy5/60.;	

	  /* check for yyyy/mm/dd */
	  if (idummy1 > 1800) {
		  (*(numeric_fields+i))[rec] =
		  days_from_day0_(&days_1900,&idummy1,&idummy2,&idummy3,&rdum,status);
		  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
	  } else {

	  /* Check for 2-digit or 4-digit year */
		  if (idummy3 < 100) { 
			  if (idummy3 < break_century)   /* assign 2-digit year to century*/
			  idummy3 += 2000;
		  else
			  idummy3 += 1900;
		  }
		  
		  (*(numeric_fields+i))[rec] =
		  days_from_day0_(&days_1900,&idummy3,&idummy1,&idummy2,&rdum,status);
		  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
	  }

	  (*(numeric_fields+i))[rec] = rdum + tpart/24.;
	  if (tpart == -999) (*(numeric_fields+i))[rec] = bad_flags[i];

	  /* force dates with dashes "-" to be in yyyy-mm-dd format *kob* */
	} else if (sscanf(p,"%4d-%2d-%2d %d:%d:%lf%1s",
			  &idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,errstr)  >= 4) {
		
		ndum = sscanf(p,"%4d-%2d-%2d %d:%d:%lf%1s",&idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,errstr);
		tpart = -999;	
		if (ndum == 6) tpart = idummy4 + idummy5/60. + dummy/3600.;
		if (ndum == 5) tpart = idummy4 + idummy5/60.;

	  (*(numeric_fields+i))[rec] =
	    days_from_day0_(&days_1900,&idummy1,&idummy2,&idummy3,&rdum,status);
	  (*(numeric_fields+i))[rec] = rdum + tpart/24.; 
	  if (tpart == -999) (*(numeric_fields+i))[rec] = bad_flags[i];
	  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
	  }

	/* check for yyyymmdd date */
	else if ( (sscanf(p,"%4d%2d%2d%1s",&idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,str1) >= 4)
	      && idummy1>0
	      && idummy3>=1 && idummy3<=12
	      && idummy2>=1 && idummy2<=31 ) {
		ndum = sscanf(p,"%4d%2d%2d%1s",&idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,str1);
		tpart = -999;	
		if (ndum == 6) tpart = idummy4 + idummy5/60. + dummy/3600.;
		if (ndum == 5) tpart = idummy4 + idummy5/60.;	

		(*(numeric_fields+i))[rec] =
		days_from_day0_(&days_1900,&idummy1,&idummy2,&idummy3,&rdum,status);
		(*(numeric_fields+i))[rec] = rdum + tpart/24.; 	
	    if (tpart == -999) (*(numeric_fields+i))[rec] = bad_flags[i];
		if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
	  }
	else
	  (*(numeric_fields+i))[rec] = bad_flags[i];
	break;
	
	/* Euro-date */
      case FTYP_EURODATE:
	if (sscanf(p,"%d/%d/%d%1s",&idummy1,&idummy2,&idummy3,errstr) == 3) {
	  /* need to check to see if idummy3 which contains the year is in 
	     the form YY or YYYY     *kob*  10/02  */
		 
	  /* Date of 00/00/00 is bad-data */
	  if (idummy1 == 0 || idummy2 == 0 || idummy3 == 0) {
		  (*(numeric_fields+i))[rec] = bad_flags[i];
		  break;
	  }

	  /* check for yyyy/mm/dd */
	  if (idummy1 > 1800) {
		  (*(numeric_fields+i))[rec] =
		  days_from_day0_(&days_1900,&idummy1,&idummy2,&idummy3,&rdum,status);
		  (*(numeric_fields+i))[rec] = rdum;
		  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
		  break;
	  }

	  /* Check for 2-digit or 4-digit year */
	  if (idummy3 < 100) { 
	    if (idummy3 < break_century)   /* assign 2-digit year to century*/
	      idummy3 += 2000;
	    else
	      idummy3 += 1900;
	  }

	  (*(numeric_fields+i))[rec] =
	    days_from_day0_(&days_1900,&idummy3,&idummy2,&idummy1,&rdum,status);
	  (*(numeric_fields+i))[rec] = rdum;
	  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
	  /* force dates with dashes "-" to be in yyyy-mm-dd format *kob* */
	} else if (sscanf(p,"%4d-%2d-%2d%1s",
			  &idummy1,&idummy2,&idummy3,errstr) == 3) {
	  (*(numeric_fields+i))[rec] =
	    days_from_day0_(&days_1900,&idummy1,&idummy2,&idummy3,&rdum,status);
	  (*(numeric_fields+i))[rec] = rdum;
	  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i]; }
	/* add check for yyyyddmm euro date *kob* */
	else if ( (sscanf(p,"%4d%2d%2d%1s",&idummy1,&idummy2,&idummy3,str1)==3)
	      && idummy1>0
	      && idummy3>=1 && idummy3<=12
	      && idummy2>=1 && idummy2<=31 ) {
	  (*(numeric_fields+i))[rec] =
	    days_from_day0_(&days_1900,&idummy1,&idummy3,&idummy2,&rdum,status);
	  (*(numeric_fields+i))[rec] = rdum;
	  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i]; }
	else
	  (*(numeric_fields+i))[rec] = bad_flags[i];
	break;

	/* Euro-date and time
	   Parses fields that are date time separated by a space.
	   It would be good to also allow for Z or T separator
	   Currently those can be handled as separate date and time
	   fields with T or Z given as a delimiter. */

	      case FTYP_EDATIME:
			  
	if (sscanf(p,"%d/%d/%d %d:%d:%lf%1s",&idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,errstr) >= 4) {

	  /* Date of 00/00/00 in any order is bad-data */
	  if (idummy1 == 0 || idummy2 == 0 || idummy3 == 0) {
		  (*(numeric_fields+i))[rec] = bad_flags[i];
		  break;
	  }

	  ndum = sscanf(p,"%d/%d/%d %d:%d:%lf%1s",&idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,errstr);
	  tpart = -999;
	  if (ndum == 6) tpart = idummy4 + idummy5/60. + dummy/3600.;
	  if (ndum == 5) tpart = idummy4 + idummy5/60.;	

	  /* check for yyyy/mm/dd */
	  if (idummy1 > 1800) {
		  (*(numeric_fields+i))[rec] =
		  days_from_day0_(&days_1900,&idummy1,&idummy2,&idummy3,&rdum,status);

	  } else {

	  /* Check for 2-digit or 4-digit year */
		  if (idummy3 < 100) { 
			  if (idummy3 < break_century)   /* assign 2-digit year to century*/
			  idummy3 += 2000;
		  else
			  idummy3 += 1900;
		  }
		  
		  (*(numeric_fields+i))[rec] =
		  days_from_day0_(&days_1900,&idummy3,&idummy2,&idummy1,&rdum,status);
	  }

	  (*(numeric_fields+i))[rec] = rdum + tpart/24.;
	  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
	  if (tpart == -999) (*(numeric_fields+i))[rec] = bad_flags[i];

	  /* force dates with dashes "-" to be in yyyy-mm-dd format *kob* */
	} else if (sscanf(p,"%4d-%2d-%2d %d:%d:%lf%1s",
			  &idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,errstr)  >= 4) {
		
		ndum = sscanf(p,"%4d-%2d-%2d %d:%d:%lf%1s",&idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,errstr);
		tpart = -999;	
		if (ndum == 6) tpart = idummy4 + idummy5/60. + dummy/3600.;
		if (ndum == 5) tpart = idummy4 + idummy5/60.;

	  (*(numeric_fields+i))[rec] =
	    days_from_day0_(&days_1900,&idummy1,&idummy2,&idummy3,&rdum,status);
	  (*(numeric_fields+i))[rec] = rdum + tpart/24.; 
	  if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];
	  if (tpart == -999) (*(numeric_fields+i))[rec] = bad_flags[i];
	  }

	/* add check for yyyyddmm euro date *kob* */
	else if ( (sscanf(p,"%4d%2d%2d%1s",&idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,str1) >= 4)
	      && idummy1>0
	      && idummy3>=1 && idummy3<=12
	      && idummy2>=1 && idummy2<=31 ) {
		ndum = sscanf(p,"%4d%2d%2d%1s",&idummy1,&idummy2,&idummy3,&idummy4,&idummy5,&dummy,str1);
		tpart = -999;	
		if (ndum == 6) tpart = idummy4 + idummy5/60. + dummy/3600.;
		if (ndum == 5) tpart = idummy4 + idummy5/60.;	

		(*(numeric_fields+i))[rec] =
		days_from_day0_(&days_1900,&idummy1,&idummy3,&idummy2,&rdum,status);
		(*(numeric_fields+i))[rec] = rdum + tpart/24.;
	    if (tpart == -999) (*(numeric_fields+i))[rec] = bad_flags[i];
	    if (*status != 3) (*(numeric_fields+i))[rec] = bad_flags[i];  }
	else
	  (*(numeric_fields+i))[rec] = bad_flags[i];
	break;
	
	/* time */
      case FTYP_TIME:
	if (sscanf(p,"%d:%d:%lf%1s",&idummy1,&idummy2,&dummy,errstr) == 3)
	  (*(numeric_fields+i))[rec] = idummy1 + idummy2/60. + dummy/3600.;
	else if (sscanf(p,"%d:%d%1s",&idummy1,&idummy2,errstr) == 2)
	  (*(numeric_fields+i))[rec] = idummy1 + idummy2/60.;
	else
	  (*(numeric_fields+i))[rec] = bad_flags[i];
	break;
	
	/* generic numeric field */
      case FTYP_NUMERIC:
	if ( sscanf(p,"%lf%1s",&dummy,errstr ) != 1)
	  (*(numeric_fields+i))[rec] = bad_flags[i];
	/* *kob* 12/01 - need to check for a sting ending in e or E
	   osf and linux compiler let such a string through as 
	   a valid number, rather than a longitude */
	else {
	  (*(numeric_fields+i))[rec] = dummy;
	  slen = strlen(p);
	  if (p[slen-1] == 'e' || p[slen-1] == 'E') 
	    (*(numeric_fields+i))[rec] = bad_flags[i];  
	}
	break;
	
	/* character field */
      case FTYP_CHARACTER:
	{
	  /* remove surrounding quotations, if any */
	  if (strlen(p)>1 && *p=='"' && *(p+strlen(p)-1)=='"') {
	    *(p+strlen(p)-1) = '\0';
	    p++;
	  }
	  (*(text_fields+i))[rec*pinc] =
	    (char *) malloc(sizeof(char)*(strlen(p)+1));
	  strcpy( (*(text_fields+i))[rec*pinc], p );
	}
	break;
	
      default:
	printf("internal error: unknown field type");
      }
    }

/* if bad status flag out of a date conversion, make the date the bad-flag
   Then the user can deal with undefined times as desired */

 	if (*status != 3)  (*(numeric_fields+i))[rec] = bad_flags[i]; 
    p = pnext;	
    }

}

/*
 *
 analRec - determine the number of fields and types of each field on one rec
	  - if subsequent records produce differing analyses of a field then
	    flag it as a CHARACTER fields
 analRec_ arguments
 recptr - buffer holding file record
 delims - string containing delimiter in file (note that blank is
          considered a special delimiter representing any number of blanks)
 nfields - (output) number of fields determined to be in file
 field_types - (output)type of each field
 max_fields - (input) maximum number of fields to consider
 *
 */

void analRec(char *recptr, char *delims, int* nfields, int field_type[],
	    int max_fields)
{

  char *p, *pnext, pstart[256], str1[2], latlon1[2];
  double dummy;

  int idummy1, idummy2, idummy3, i, nfields_in;

  p = recptr;
  nfields_in = *nfields;
  *nfields = 0;

  /* if information is available about a field for the first time, apply it
     if the analysis differs from a previous then call it a character field */
  while (p  != NULL) {
    pnext = nexstrtok(p, delims);
    if (*p == '\0')
      /* null field like 2 commas in a row */
      {
	/* retain previous information */
      } 
    else if (sscanf(p,"%d/%d/%d%1s",&idummy1,&idummy2,&idummy3,str1) == 3)
      /* date as mm/dd/yy */
      {
	if (field_type[(*nfields)] == FTYP_MISSING)
	  field_type[(*nfields)] = FTYP_DATE;
	else if (field_type[(*nfields)] != FTYP_DATE)
	  field_type[(*nfields)] = FTYP_CHARACTER;
      }
    else if (sscanf(p,"%d-%d-%d%1s",&idummy1,&idummy2,&idummy3,str1) == 3)
      /* date as yyyy-mm-dd */
      {
	if (field_type[(*nfields)] == FTYP_MISSING)
	  field_type[(*nfields)] = FTYP_DATE;
	else if (field_type[(*nfields)] != FTYP_DATE)
	  field_type[(*nfields)] = FTYP_CHARACTER;
      }
    else if ( (sscanf(p,"%4d%2d%2d%1s",&idummy1,&idummy2,&idummy3,str1) == 3)
	      && idummy1>=1800 && idummy1<2100
	      && idummy2>=1 && idummy2<=12
	      && idummy3>=1 && idummy3<=31 )
      /* date as yyyymmdd */
      {
	if (field_type[(*nfields)] == FTYP_MISSING)
	  field_type[(*nfields)] = FTYP_DATE;
	else if (field_type[(*nfields)] != FTYP_DATE)
	  field_type[(*nfields)] = FTYP_CHARACTER;
      }
    else if (sscanf(p,"%d:%d:%lf%1s",&idummy1,&idummy2,&dummy,str1) == 3)
      /* time as hh:mm:ss.s */
      {
	if (field_type[(*nfields)] == FTYP_MISSING)
	  field_type[(*nfields)] = FTYP_TIME;
	else if (field_type[(*nfields)] != FTYP_TIME)
	  field_type[(*nfields)] = FTYP_CHARACTER;
      }
    else if (sscanf(p,"%d:%d%1s",&idummy1,&idummy2,str1) == 2)
      /* time as hh:mm */
      {
	if (field_type[(*nfields)] == FTYP_MISSING)
	  field_type[(*nfields)] = FTYP_TIME;
	else if (field_type[(*nfields)] != FTYP_TIME)
	  field_type[(*nfields)] = FTYP_CHARACTER;
      }
    else if (sscanf(p,"%lf%1[NnSs]%1s",&dummy,latlon1,str1) == 2
	     && dummy>-90.1 && dummy<90.1 )
      /* latitude */
      {
	if (field_type[(*nfields)] == FTYP_MISSING
	    || field_type[(*nfields)] == FTYP_NUMERIC)
	  field_type[(*nfields)] = FTYP_LAT;
	else if (field_type[(*nfields)] != FTYP_LAT)
	  field_type[(*nfields)] = FTYP_CHARACTER;
      }
    else if ((sscanf(p,"%lf%1[EeWw]%1s",&dummy,latlon1,str1) == 2) 
	     || p[(strlen(p))-1] == 'E' || p[(strlen(p))-1] == 'E' )
      /* need above check for strings such as 120E - which some 
	 compilers pass through as valid floating point number 
         kob v5.41 - 4/02 */
      /* But, fixing bug 1700, check whether the rest of the field
	  is numeric before calling this Longitude. (The string ZAIRE
	  was being marked as Longitude) 
         acm v6.31 - 10/09 */
       /* longitude */
      {
	if (field_type[(*nfields)] == FTYP_MISSING
	    || field_type[(*nfields)] == FTYP_NUMERIC)
	  field_type[(*nfields)] = FTYP_LON;
	else if (field_type[(*nfields)] != FTYP_LON)
	  field_type[(*nfields)] = FTYP_CHARACTER;
	  idummy1 = (strlen(p))-1;
	  strncpy(pstart,p,idummy1 );
	if (sscanf(pstart,"%lf",&dummy) != 1)
	  field_type[(*nfields)] = FTYP_CHARACTER;
      }
    else if (sscanf(p,"%lf%1s",&dummy,str1) == 2)
      /* digits followed by trash -- not a legal numeric field*/
      {
	field_type[(*nfields)] = FTYP_CHARACTER;
      }
    else if (sscanf(p,"%lf",&dummy) == 1)
      /* numeric field */
      /* note that pure numeric fields may be lats or longs */
      {
	if (field_type[(*nfields)] == FTYP_MISSING)
	  field_type[(*nfields)] = FTYP_NUMERIC;
	else if (field_type[(*nfields)] == FTYP_LAT
		 && dummy>-90.1 && dummy<90.1 )
	  field_type[(*nfields)] = FTYP_LAT;
	else if (field_type[(*nfields)] == FTYP_LON)
	  field_type[(*nfields)] = FTYP_LON;
	else if (field_type[(*nfields)] != FTYP_NUMERIC)
	  field_type[(*nfields)] = FTYP_CHARACTER;
      }
    else
      /* any other text */
      field_type[(*nfields)] = FTYP_CHARACTER;
      
#ifdef diagnostic_output	  /* ************* */
    printf("%d %s:  %d\n", *nfields, p, field_type[(*nfields)]);
#endif                            /* ************* */

    if ( *nfields < max_fields )
      {    
	(*nfields)++;
	p = pnext;
      }
    else
      break;  /* cannot look at all of the fields */

    }

  /* if records have unequal length, return the longest found */
  *nfields = *nfields > nfields_in ? *nfields : nfields_in;

  return;
}

/*
 *
 *
 */

char *nexstrtok(char *s1, char *s2)

     /*
       like strtok but sensitive to multiple (non-white space) delimiters
       as significant. For example, 2 commas together indicate a missing field.
     */

     /* note - this routine will modify the s1 string */
{
  char *p2, *nex;

  /* sanity check that we have a valid input record */
  if (s1 == NULL)
    return NULL;

  /* find the next delimiter */
  p2 = strpbrk( s1, s2 );

  if (p2 == NULL)
    return NULL;
  else
    {
      nex = p2 + 1;

      /* skip trailing blanks in this field */
      while (*(p2-1)==' ')
	p2--;
      *p2 = '\0';

      /* Skip leading blanks in next field */
      while (*nex==' ')
	nex++;

      return nex;
    }
}

/*
 *
 save_delimited_info - allocate struct memoro and save special info needed
                       for delimited file reads
	nfields - number of fields (variables) in file
	field_type - field types for each variable
	delim - list of delimiters to use when reading the file
	ptr - returned pointer to structure
 *
 */
void FORTRAN(save_delimited_info) (int *nfields, int field_type[],
				   char *delim, DelimFileInfo **ptr)
{
  DelimFileInfo *fi = (DelimFileInfo *) calloc(1, sizeof(DelimFileInfo));
  int* _field_type  = (int *) malloc(sizeof(int) * (*nfields));
  char* _delim      = (char *) malloc(sizeof(char) * (int)strlen(delim));
  int i;

  for (i=0; i<*nfields; i++)
    _field_type[i] = field_type[i];

  strcpy(_delim, delim);

  fi->nfields = *nfields;
  fi->fieldType = _field_type;
  fi->delim = _delim;

  *ptr = fi;
  return;
}

void FORTRAN(get_delimited_info) (int *nfields, int field_type[],
				   char *delim, DelimFileInfo **ptr)
{

  int i,iout;
  DelimFileInfo *fi = *ptr;

  *nfields = fi->nfields;
  for (i=0; i<*nfields; i++)
    field_type[i] = (fi->fieldType)[i];
  strcpy(delim, fi->delim);
  return;
}

void FORTRAN(delete_delimited_info) (DelimFileInfo **ptr)
{
  DelimFileInfo *fi = *ptr;
  free(fi->fieldType);
  free(fi->delim);
  free(fi);
  return;
}

