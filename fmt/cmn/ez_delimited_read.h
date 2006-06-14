/* delimitedRead.h
 *
 * Steve Hankin
 * October 2000
 *
 * Header file information needed for spreadsheet-style delimited reads
 * by the Ferret program
 * v600 *acm* change call to days_from_da0 needed for 64-bit build
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

#ifdef NO_ENTRY_NAME_UNDERSCORES
#define FORTRAN(a) a
#else
#define FORTRAN(a) a##_
#endif

typedef struct _DelimFileInfo {
  int nfields;              /* number of variables to read in this file */
  int *fieldType;           /* field type of each variable */
  char *delim;              /* character array of delimiters */
} DelimFileInfo;


char *nexstrtok(char *s1, char *s2);
void analRec(char *recptr, char *delims, int* nfields, int field_type[],
	    int max_fields);
int decodeRec(char *recptr, char *delims, int* nfields, int field_type[],
	      int nrec,
	      float** numeric_fields, char*** text_fields, float bad_flags[]);
int decode_file (char* fname, char *recptr, char *delims, int* skip,
	       int *maxrec, int* reclen, int* nfields, int field_type[],
	       int* nrec,
	       float** numeric_fields, char*** text_fields, float bad_flags[]);

float FORTRAN(days_from_day0) (double *days1900, int* iyr, int* imon, int* iday, float* rdum);

void FORTRAN(decode_file_jacket)
		( char* fname, char *recptr, char *delims, int *skip,
		  int* maxrec, int* reclen, int* nfields,
		  int field_type[], int* nrec,
		  int mrlist[], float *memptr, int mr_blk1[], int* mblk_size,
		  float mr_bad_flags[], char ***mr_c_ptr);
int FORTRAN(anal_file) (char *fname, char *recptr, char *delims, int *skip,
	     int *maxrec, int* reclen, int* nfields, int field_type[],
	     int *max_fields);
void FORTRAN(save_delimited_info) (int *nfields, int field_type[],
				   char *delims, DelimFileInfo **ptr);
void FORTRAN(get_delimited_info) (int *nfields, int field_type[],
				   char *delim, DelimFileInfo **ptr);
void FORTRAN(delete_delimited_info) (DelimFileInfo **ptr);
