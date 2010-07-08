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
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import thredds.catalog.InvCatalogImpl;
import thredds.catalog.InvDataset;
import thredds.catalog.InvDatasetImpl;
import thredds.catalog.InvService;
import thredds.catalog.ServiceType;

/**
 * Scanner for a local directory to generate a complete catalog of the tree.
 * @author Karl M. Smith - karl.smith (at) noaa.gov
 */
public class LocalDirTreeScanner {

	/** root of the local directory tree to scan */
	private File localDir;

	/**
	 * Create a scanner ready to scan the local directory tree rooted at localDir
	 * @param localDir local root directory of the tree to scan
	 * @throws IOException if localDir does not exist or is not a directory
	 */
	public LocalDirTreeScanner(File localDir) throws IOException {
		if ( ! localDir.isDirectory() )
			throw new IOException(localDir.getPath() + " is not a valid local directory");
		this.localDir = localDir;
	}

	/**
	 * Generate a complete catalog of the local directory tree rooted at this 
	 * @param datasetFilter filter for the files/directories added to the catalog; 
	 * can be null, in which case all files/directories are added
	 * @return a complete catalog of the directory tree rooted at the local directory 
	 * used in the construction of this class 
	 * @throws IOException if any of the file system operations throws one or 
	 * if the local directory used in the construction of this class is unable to be read
	 */
	public InvCatalogImpl generateCatalog(FileFilter datasetFilter) throws IOException {
	    // Create the service for the catalog and datasets
	    InvService service = new InvService("file:", ServiceType.FILE.toString(), "file:", null, null);

	    // Create a new catalog
		InvCatalogImpl catalog = new InvCatalogImpl(localDir.getPath(), null, localDir.toURI());
		catalog.addService(service);

		// Create the top-level dataset.  Since an access will be created in finish() 
		// for each dataset with a URI path, don't add URI paths to directories
		InvDatasetImpl topDataset = new LocalDirInvDatasetImpl(null, localDir.getName(), service.getName());
		topDataset.setID(localDir.getPath());
		topDataset.setCatalog(catalog);

		// Recurse into the tree, adding InvDatasetImpl objects to the datasets field
		addContentDatasets(topDataset, localDir, datasetFilter, service);

		// Add this dataset to the catalog
		catalog.addDataset(topDataset);

		if ( ! catalog.finish() )
			throw new IOException("Unable to finish construction of the catalog from " + localDir.getPath());
		return catalog;
	}

	/**
	 * Adds datasets to parentDataset for each file/directory in parentDir that passes the filter.
	 * When subdirectories are encountered, this method calls itself with the subdirectory dataset
	 * and subdirectory File, thus filling out the full directory tree rooted at parentDir.
	 * @param parentDataset add new child datasets to this dataset
	 * @param parentDir directory to examine
	 * @param datasetFilter filter on the files/directories to be added to the catalog.  
	 * If null, all files/directories are added.
	 * @param serviceName service name to be added to the InvDatasetImpl objects created
	 * @throws IOException if any of the file system operations throws one or if parentDir 
	 * is unable to be read
	 */
	private void addContentDatasets(InvDatasetImpl parentDataset, File parentDir, FileFilter datasetFilter, InvService service) throws IOException {
		// Get the list of files and directories in this directory
		File[] contentsArray = parentDir.listFiles(datasetFilter);
		if ( contentsArray == null )
			throw new IOException("Unable to list the contents of " + parentDir.getPath());

		// Sort the array of files and directories
		if ( contentsArray.length > 1 ) {
			List<File> contentsList = Arrays.asList(contentsArray);
			Collections.sort(contentsList);
		}

		// Add to the parent's datasets array a dataset for each file/dir returned
		List<InvDataset> datasets = parentDataset.getDatasets();
		for (File child : contentsArray) {
			if ( child.isDirectory() ) {
				// Create the dataset with a null urlPath argument so no access created
				InvDatasetImpl childDataset = new LocalDirInvDatasetImpl(parentDataset, child.getName(), service.getName());
				childDataset.setID(child.getPath());
				addContentDatasets(childDataset, child, datasetFilter, service);
				datasets.add(childDataset);
			}
			else {
				// Get the URI string
				String uriPathString = child.toURI().toString();
				// Remove the base string from the service
				String serviceBaseString = service.getBase();
				if ( ! uriPathString.startsWith(serviceBaseString) )
					throw new IOException("Unexpected URI string of child: " + uriPathString + "\n" +
										  "under service URI string: " + serviceBaseString);
				uriPathString = uriPathString.substring(serviceBaseString.length());
				// Create the dataset with this urlPath argument so an appropriate access is created
				InvDatasetImpl childDataset = new InvDatasetImpl(parentDataset, child.getName(), null, service.getName(), uriPathString);
				childDataset.setID(child.getPath());
				childDataset.setDataSize(child.length());
				childDataset.setLastModifiedDate(new Date(child.lastModified()));
				datasets.add(childDataset);
			}
		}
	}

}
