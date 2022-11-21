#!/usr/bin/env python
'''

Copyright (C) 2018 Vanessa Sochat.

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
License for more details.

See <https://www.gnu.org/licenses/> for a copy of the GNU Affero 
General Public License

'''

# Can we upload sample data to the Kaggle API, linked with a dinosaur dataset?
# https://github.com/Kaggle/kaggle-api

from kaggle.api import KaggleApi
from datetime import datetime
import tempfile
import argparse
import shutil
import json
import sys
import os


def get_parser():
    description = "Dinosaur Kaggle Dataseet Creator"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--username", dest='username', 
                        help="the kaggle username (in the url)", 
                        type=str, default=None, required=True)


    parser.add_argument("--title", "-t", dest='title', 
                        help="verbose title for the dataset", 
                        type=str, default=None, required=True)

    parser.add_argument("--name", dest='name', 
                        help="The identifier (name) of the dataset", 
                        type=str, default=None)

    parser.add_argument("--keywords",'-k', dest='keywords', 
                        help="comma separated (no spaces) keywords for dataset", 
                        type=str, default=None)

    parser.add_argument("--files",'-f', nargs='*', dest='files',
                        help="dataset files", 
                        type=str)

    parser.add_argument("--license",'-l', dest='license', 
                        help="license name for file description, uses 'other' in metadata", 
                        type=str, default='MIT License')

    return parser



def main():
    '''the main entrypoint for pushing!
    '''

    parser = get_parser()

    # If the user didn't provide any arguments, show the full help
    if len(sys.argv) == 1:
        parser.print_help()
    try:
        args = parser.parse_args()
    except:
        sys.exit(0)

    # If a name isn't provided, derive from title
    name = args.name
    if name is None:
        name = args.title.replace(' ','-').lower()

    # Pass on to the correct parser
    return_code = 0
    try:
        create_dataset(keywords=args.keywords, 
                       username=args.username,
                       title=args.title,
                       files=args.files,
                       license=args.license,
                       name=name)

        sys.exit(return_code)
    except UnboundLocalError:
        return_code = 1



def create_dataset(keywords, username, title, files, license, name):
    '''create a Kaggle dataset for a given set of files.
       The description should be done in the web interface.

       Parameters
       ==========
       username: your kaggle username, or the name of an organization that the dataset will belong to
       title: the title to give the dataset (put in quotes if you have spaces)
       name: the name of the dataset itself (no spaces or special characters, and good practice to put in quotes)
       keywords: comma separated list of keywords (no spaces!)
       files: full paths to the data files to upload

    '''

    # keywords likely come in as one,two,three
    if keywords is not None:
        if not isinstance(keywords, list):
            keywords = keywords.split(',')

    # create an API client
    client = KaggleApi()

    # Create a new metadata file
    # kaggle datasets init -p /path/to/dataset
    # https://github.com/Kaggle/kaggle-api/wiki/Metadata
    workdir = tempfile.mkdtemp()

    # The function doesn't return the filename, but it's predictable
    meta_file = '/%s/datapackage.json' % workdir
    client.dataset_initialize(workdir)

    # Update the metadata json
    with open(meta_file,'r') as fh:
        content = json.load(fh)

    content['title'] = title
    content['id'] = '%s/%s' %(username, name)
    content['licenses'] = [{'name':'other'}]
    if keywords is not None:
        content['keywords'] = keywords
    content['resources'] = []

    print('Preparing upload folder...')

    # Generate date descriptor
    today = datetime.today()

    # Add each resource file
    for datafile in files:
        filename = os.path.basename(datafile)
        copyfile = '%s/%s' %(workdir, filename)
        shutil.copyfile(datafile, copyfile)
        content['resources'].append({
          "path": filename,
          "description": "%s, part of %s Dataset, %s/%s, %s" %(filename, 
                                                               title,
                                                               today.month,
                                                               today.year,
                                                               license)
        })

    # Show for the user
    print(json.dumps(content, indent=1))

    # Update the metadata file
    with open(meta_file,'w') as fh:
        json.dump(content, fh)

    client.authenticate()
    result = client.dataset_create_new(workdir,
                                       convert_to_csv=False)

    print('The following URL will be available after processing (10-15 minutes)')
    print(result)

if __name__ == '__main__':
    main()