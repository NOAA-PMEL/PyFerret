#ifndef _EZ_DELIMITED_READ_H_
#define _EZ_DELIMITED_READ_H_

/* delimitedRead.h
 *
 * Steve Hankin
 * October 2000
 *
 * Header file information needed for spreadsheet-style delimited reads
 * by the Ferret program
 * v600 *acm* change call to days_from_day0 needed for 64-bit build
 *
 *  *acm*  1/12      - Ferret 6.8 ifdef double_p for double-precision ferret.

  V701 7/16 *acm*   ticket 2450: add date-time and euro-date-time field types
                    ticket 2449: report incorrect choice of date/ eurodate as an error
 */

#define FTYP_MISSING 1
#define FTYP_NUMERIC 2
#define FTYP_CHARACTER 3
#define FTYP_LAT 4
#define FTYP_LON 5
#define FTYP_DATE 6
#define FTYP_EURODATE 7
#define FTYP_TIME 8
#define FTYP_DATIME 9
#define FTYP_EDATIME 10

#define FANAL_OK 0
#define FANAL_HAS_MISSING 1


typedef struct _DelimFileInfo {
  int nfields;              /* number of variables to read in this file */
  int *fieldType;           /* field type of each variable */
  char *delim;              /* character array of delimiters */
} DelimFileInfo;


/* prototypes that use DelimFileInfo */
void FORTRAN(save_delimited_info)(int *nfields, int field_type[], char *delims, DelimFileInfo **ptr);
void FORTRAN(get_delimited_info)(int *nfields, int field_type[], char *delim, DelimFileInfo **ptr);
void FORTRAN(delete_delimited_info)(DelimFileInfo **ptr);

#endif
