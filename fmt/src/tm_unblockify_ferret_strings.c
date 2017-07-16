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
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE.  
*/

/* 
    The variable mr_blk1 is an array of pointers to strings
    (to be allocated and assigned) where the string pointers
    are spaced 8 bytes apart.

    The variable pblock is a bufsiz-long character array of
    string data where each string uses filestrlen characters
    and is null terminated if shorter than filestrlen
    characters.

    This function copies the strings from pblock into newly 
    allocated strings and assigns them to mr_blk1.
*/

#include <Python.h> /* make sure Python.h is first */
#include <stdlib.h>

void tm_unblockify_ferret_strings(char **mr_blk1, char *pblock,
				  int bufsiz, int filestrlen)
{
  int i, n;
  char *pinchar, *pinstr, *poutchar, **poutstr;

  poutstr = mr_blk1;  /* points to each output string pointer in turn */
  pinstr  = pblock;   /* points to each input string in turn */

  for (i=0; i<bufsiz/filestrlen; i++) {

    /* measure the length of the string */
    n = 0;
    while ( (n < filestrlen) && (pinstr[n] != '\0') )
       n++;

    /* save the pointer to the first character in this string */
    pinchar = pinstr;

    /* increment pinstr to point to the next input string */
    pinstr += filestrlen;

    /* allocate memory for this string */
    poutchar = (char *) PyMem_Malloc(sizeof(char) * (n+1));

    /*
     * Free any existing string in the output array and then
     * assign this newly allocated memory to the output array.
     */
    if ( *poutstr != NULL )
       PyMem_Free(*poutstr);
    *poutstr = poutchar;

    /* increment poutstr to point to the next output string pointer position */
    poutstr += (8/sizeof(char**));

    /* copy this input string into the newly allocated memory */
    n = 0;
    while ( (n < filestrlen) && (*pinchar != '\0') ) {
      *poutchar = *pinchar;
      poutchar++;
      pinchar++;
      n++;
    }
    *poutchar = '\0';

  }
}
