/* delimitedRead.h
 *
 * Steve Hankin
 * October 2000
 *
 * Header file information needed for spreadsheet-style delimited reads
 * by the Ferret program
 *
 */

#define FTYP_MISSING 1
#define FTYP_NUMERIC 2
#define FTYP_CHARACTER 3
#define FTYP_LAT 4
#define FTYP_LON 5
#define FTYP_DATE 6
#define FTYP_EURODATE 7
#define FTYP_TIME 8

#define FANAL_OK 0
#define FANAL_HAS_MISSING 1


char *nexstrtok(char *s1, char *s2);
void analRec(char *recptr, char *delims, int* nfields, int field_type[],
	    int max_fields);
int analFile(char *fname, char *recptr, char *delims, int *skip,
	     int *maxrec, int* reclen, int* nfields, int field_type[],
	     int max_fields);
int decodeRec(char *recptr, char *delims, int* nfields, int field_type[],
	      int nrec,
	      float** numeric_fields, char*** text_fields, float bad_flags[]);
int decodeFile(char* fname, char *recptr, char *delims, int* skip,
	       int *maxrec, int* reclen, int* nfields, int field_type[],
	       int* nrec,
	       float** numeric_fields, char*** text_fields, float bad_flags[]);

float days_from_day0_(double* days_1900, int* iyr, int* imon, int* iday);





