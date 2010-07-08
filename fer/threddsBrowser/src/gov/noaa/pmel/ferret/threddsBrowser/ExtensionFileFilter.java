/**
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
package gov.noaa.pmel.ferret.threddsBrowser;

import java.io.File;
import java.io.FileFilter;
import java.util.Collection;
import java.util.HashSet;

/**
 * A file filter based on a set of filename extensions
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class ExtensionFileFilter extends HashSet<String> implements FileFilter {

	private static final long serialVersionUID = -5282011356208993578L;

	/**
	 * Create a file filter with an empty set of extensions; thus
	 * all files are accepted.
	 */
	public ExtensionFileFilter() {
	}

	/**
	 * Create a file filter from the given Collection of extensions.
	 * If this set of extensions is empty, all files are accepted.
	 * @param extensionsSet the set of acceptable extensions. Cannot be null.
	 * The extensions should <b> not </b> contain the '.'
	 */
	public ExtensionFileFilter(Collection<String> extensionsSet) {
		super(extensionsSet);
	}

	/**
	 * @return if fileToCheck is a directories or if it is a file with an extension 
	 * given in the set of extension (if not empty) used to construct this object.
	 * (If that set of extensions is empty, all files are accepted.)
	 */
	@Override
	public boolean accept(File fileToCheck) {
		if ( size() == 0 )
			return true;
		if ( fileToCheck.isDirectory() )
			return true;
		String name = fileToCheck.getName();
		int loc = name.lastIndexOf('.') + 1;
		if ( (loc <= 0) || (loc >= name.length()) )
			return false;
		String ext = name.substring(loc);
		if ( contains(ext) )
			return true;
		return false;
	}

}
