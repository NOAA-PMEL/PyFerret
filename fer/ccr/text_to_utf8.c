/*
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

#include <Python.h> /* make sure Python.h is first */
#include "utf8str.h"

/*
 * Translates an extended-character string into a UTF-8 string.
 *
 * Input: 
 *     text - string that include extended characters (> 0x7F)
 *     textlen - length of text
 * Output:
 *     utf8str - null-terminated UTF-8 string representation of text;
 *               every extended character in text will result 
 *               in two characters in this array (so ideally,
 *               has minimum length 2*textlen + 1)
 *     utf8strlen - length of the UTF-8 string created
 */
void text_to_utf8_(const char *text, const int *textlen, char *utf8str, int *utf8strlen)
{
    int codept;
    char utf8chars[4];
    int  numutf8chars;
    int  j, k, q;

    for (j = 0, k = 0; j < *textlen; j++) {
        codept = (unsigned char) (text[j]);
        create_utf8_str_(&codept, utf8chars, &numutf8chars);
        for (q = 0; q < numutf8chars; q++, k++)
           utf8str[k] = utf8chars[q];
    }
    utf8str[k] = '\0';
    *utf8strlen = k;
}

