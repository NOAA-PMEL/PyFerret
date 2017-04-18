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
#include "ferret.h"

/*
 * Assigns a character string with the values for a UTF-8 character
 *
 * Input: 
 *     codepoint - the UTF-8 code point (integer) value
 * Output:
 *     utf8str (array of at least four characters) - the character 
 *                  representation of the UTF-8 character
 *     utf8strlen - the number of characters in the character 
 *                  representation of the UTF-8 characters, 
 *                  or zero if the codepoint is invalid for UTF-8
 */
void FORTRAN(create_utf8_str)(const int *codepoint, char *utf8str, int *utf8strlen) 
{
    int codept = *codepoint;

    if ( codept <= 0x7F ) {
        utf8str[0] = (char) codept;
        *utf8strlen = 1;
    } 
    else if ( codept <= 0x7FF ) {
        utf8str[0] = (char) ((codept >> 6) + 0xC0);
        utf8str[1] = (char) ((codept & 0x3F) + 0x80);
        *utf8strlen = 2;
    } 
    else if (codept <= 0xFFFF) {
        utf8str[0] = (char) ((codept >> 12) + 0xE0);
        utf8str[1] = (char) (((codept >> 6) & 0x3F) + 0x80);
        utf8str[2] = (char) ((codept & 0x3F) + 0x80);
        *utf8strlen = 3;
    } 
    else if (codept <= 0x10FFFF) {
        utf8str[0] = (char) ((codept >> 18) + 0xF0);
        utf8str[1] = (((codept >> 12) & 0x3F) + 0x80);
        utf8str[2] = (((codept >> 6) & 0x3F) + 0x80);
        utf8str[3] = ((codept & 0x3F) + 0x80);
        *utf8strlen = 4;
    } 
    else {
        *utf8strlen = 0;
    }
}

