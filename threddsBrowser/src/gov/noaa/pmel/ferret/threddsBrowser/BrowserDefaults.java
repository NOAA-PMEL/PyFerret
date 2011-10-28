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

import java.awt.Dimension;
import java.io.File;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ucar.util.prefs.PreferencesExt;

/**
 * Default settings for a ThreddsBrowser
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class BrowserDefaults {

	/** XMLStore key for a Dimension giving the size of the browser JPanel */
	public static final String BROWSER_SIZE = "BrowserSize";

	/** XMLStore key for an int giving the location of the divider between the tree viewer and the HTML viewer */
	public static final String DIVIDER_LOCATION = "DividerLocation";

	/** XMLStore key for a String giving a space/comma/semicolon separated list of acceptable filename extensions for local datasets */ 
	public static final String DATASET_FILENAME_EXTENSIONS = "DatasetFilenameExtensions";

	/** XMLStore key for a String given the default directory for the local directory file chooser */
	public static final String LOCAL_BROWSE_DIRNAME = "LocalBrowseDirname";

	/** XMLStore key for a List of Strings giving locations for the drop-down list */
	public static final String LOCATION_STRINGS_LIST = "LocationStringsList";

	/** Initial Strings for the drop-down list of locations */
	private Collection<String> locations;

	/** Location of the split pane divider */
	private int dividerLocation;

	/** Initial directory of the local browse file chooser */ 
	private File localBrowseDir;

	/** String of acceptable filename extensions for datasets */
	private String extensionsString;

	/** Size of the ThreddsBrowser */
	private Dimension browserSize;
	
	/**
	 * Get the defaults for ThreddsBrowser from the given preferences and the initial
	 * locations environment variable.
	 * A location String will appear only once in the locations collection and 
	 * will be in the order when they first appear:
	 * <ol>
	 * <li> in the locations list saved in stored preferences </li>
	 * <li> in the value of the locations environment variable</li>
	 * </ol>
	 * @param prefs the stored preferences; may be null
	 * @param defLocsEnvName the name of the locations environment variable whose
	 * value is a space-separated list of possibly-quoted locations; may be null
	 */
	public BrowserDefaults(PreferencesExt prefs, String defLocsEnvName) {
		// Get the list from the store preferences
		List<String> locationsList = null;
		if ( prefs != null ) {
			@SuppressWarnings("unchecked")
			List<String> locsList = (List<String>) prefs.getList(LOCATION_STRINGS_LIST, null);
			locationsList = locsList;
		}

		// Use a LinkedHashSet to maintain ordered single copies of the locations
		locations = new LinkedHashSet<String>();
		if ( locationsList != null ) {
			for (String loc : locationsList) {
				if ( (loc != null) && ! loc.isEmpty() ) {
					locations.add(loc);
				}
			}
		}

		// Add locations from the environment variable
		try {
			String locsEnvValue = System.getenv(defLocsEnvName);
			// space separated, possibly double- or single-quoted, values - does not recognize escaping
			String patStr = "\\s*\"([^\"]*?)\"|\\s*'([^']*?)'|\\s*(\\S*)";
			Pattern p = Pattern.compile(patStr);
			Matcher m = p.matcher(locsEnvValue);
			while ( m.find() ) {
				String loc = m.group(1);
				if ( loc == null )
					loc = m.group(2);
				if ( loc == null )
					loc = m.group(3);
				if ( (loc != null) && ! loc.isEmpty() )
					locations.add(loc);
			}
		} catch (Exception e) {
			; // nothing to add
		}

		// Location of the split pane divider 
		dividerLocation = 400;
		if ( prefs != null ) {
			int loc = prefs.getInt(DIVIDER_LOCATION, dividerLocation);
			if ( loc > 0 ) {
				dividerLocation = loc;
			}
		}

		// Local browse directory
		localBrowseDir = null;
		if ( prefs != null ) {
			String localBrowseDirname = (String) prefs.getBean(LOCAL_BROWSE_DIRNAME, null);
			if ( localBrowseDirname != null ) {
				File localDir = new File(localBrowseDirname + File.separator);
				if ( localDir.isDirectory() ) {
					localBrowseDir = localDir;
				}
			}
		}

		// String of acceptable filename extensions for displayed datasets
		extensionsString = "cdf, nc, ncd, des, dat, txt";
		if ( prefs != null ) {
			String newStr = (String) prefs.getBean(DATASET_FILENAME_EXTENSIONS, extensionsString);
			// Make sure the string is in a standard, clean format
			extensionsString = createExtensionsString(parseExtensionsString(newStr));
		}

		// Size of the browser
		browserSize = new Dimension(800, 500);
		if ( prefs != null ) {
	    	Dimension size = (Dimension) prefs.getBean(BROWSER_SIZE, browserSize);
	    	if ( (size.getHeight() > 0.0) && (size.getWidth() > 0.0) ) {
	    		browserSize = size;
	    	}
		}
	}

	/**
	 * @return the complete Collection of Strings for the drop-down list of locations 
	 */
	public Collection<String> getLocationStrings() {
		return locations;
	}

	/**
	 * @return the location of the split pane divider
	 */
	public int getDividerLocation() {
		return dividerLocation;
	}

	/**
	 * @return the initial directory of the local file chooser
	 */
	public File getLocalBrowseDir() {
		return localBrowseDir;
	}

	/**
	 * @return the String of acceptable filename extensions for datasets
	 */
	public String getExtensionsString() {
		return extensionsString;
	}

	/**
	 * @return the preferred size for the ThreddsBrowser
	 */
	public Dimension getBrowserSize() {
		return browserSize;
	}

	/**
	 * Save the initial list of locations string to a PreferencesExt that can be used
	 * in the construction of a BrowserDefaults object.
	 * @param prefs save the List of location strings in here.
	 * @param locationsList the List of location strings to save; cannot be null.
	 */
	public static void saveLocationsList(PreferencesExt prefs, List<String> locationsList) {
		prefs.putList(LOCATION_STRINGS_LIST, locationsList);
	}

	/**
	 * Save the split pane divider location to a PerferencesExt that can be used
	 * in the construction of a BrowserDefaults object.
	 * @param prefs save the divider location here.
	 * @param dividerLoc the divider location to save.
	 */
	public static void saveDividerLocation(PreferencesExt prefs, int dividerLoc) {
		prefs.putInt(DIVIDER_LOCATION, dividerLoc);
	}

	/**
	 *
	 * Save the initial directory for the local file chooser to a PerferencesExt 
	 * that can be used in the construction of a BrowserDefaults object.
	 * @param prefs save the local directory here.
	 * @param localDir the local directory to save; can be null.
	 */
	public static void saveLocalBrowseDir(PreferencesExt prefs, File localDir) {
		if ( localDir != null ) {
			prefs.putBeanObject(LOCAL_BROWSE_DIRNAME, localDir.getPath());
		}
	}

	/**
	 * Save the String of acceptable filename extensions for datasets to a PreferencesExt
	 * that can be used in the construction of a BrowserDefaults object.
	 * @param prefs save the filename extensions String here.
	 * @param extsString the filename extensions String to save; cannot be null.
	 */
	public static void saveExtensionsString(PreferencesExt prefs, String extsString) {
		prefs.putBeanObject(DATASET_FILENAME_EXTENSIONS, extsString);
	}

	/**
	 * Save the size of the browser to a PreferencesExt
	 * that can be used in the construction of a BrowserDefaults object.
	 * @param prefs save the browser size here.
	 * @param size the browser size to save; cannot be null.
	 */
	public static void saveBrowserSize(PreferencesExt prefs, Dimension size) {
		prefs.putBeanObject(BROWSER_SIZE, size);
	}

	/**
	 * Parses a String of comma/semicolon/space-separated acceptable filename extensions
	 * for datasets and produces a Collection of the individual extension Strings.
	 * @param extsString the String of comma/semicolon/space-separated extensions; cannot be null.
	 * @return a Collection of the acceptable filename extension Strings.  
	 * Will not be null but may be empty.
	 */
	public static Collection<String> parseExtensionsString(String extsString) {
		// Parse the returned string to get the individual extensions
		String[] extensions = extsString.split("\\s*[\\s;,]\\s*");
		LinkedHashSet<String> extsSet = new LinkedHashSet<String>(extensions.length);
		for (String ext : extensions) {
			if ( ! ext.isEmpty() ) {
				extsSet.add(ext);
			}
		}
		return extsSet;
	}

	/**
	 * Creates String of comma-space separated acceptable filename extensions for 
	 * datasets from a Collection of individual extension Strings.
	 * @param extsColl the Collection of individual extension Strings; cannot be null.
	 * @return the String of comma-space separated acceptable filename extensions.
	 * Will not be null but may be empty.
	 */
	public static String createExtensionsString(Collection<String> extsColl) {
		StringBuilder extBuilder = new StringBuilder();
		boolean first = true;
		for (String ext : extsColl) {
			if ( (ext == null) || ext.isEmpty() )
				continue;
			if ( first )
				first = false;
			else
				extBuilder.append(", ");
			extBuilder.append(ext);
		}
		return extBuilder.toString();
	}

}
