/* Function called by tm_number to test if a string is a valid number
 * Returns 0 if not OK, 1 if OK
 *
 * J Davison 11.8.94
 */

#ifdef _NO_PROTO
#  ifdef NO_ENTRY_NAME_UNDERSCORES
void tm_number_sub  (string, result) 
#  else
void tm_number_sub_ (string, result) 
#  endif
char * string;
int * result;
#else /* NO_PROTO */
#  ifdef NO_ENTRY_NAME_UNDERSCORES
void tm_number_sub  (char * string, int * result) 
#  else
void tm_number_sub_ (char * string, int * result) 
#  endif
#endif  /* NO_PROTO */
{
  int num_read;
  float rval;
  char kval[255];

  num_read = sscanf(string, "%g%s", &rval, kval);
 
  if (num_read == 1) 
    *result = 1;
  else
    *result = 0;

  return;
}


