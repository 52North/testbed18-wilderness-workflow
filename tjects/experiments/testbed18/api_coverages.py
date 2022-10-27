import json
import requests
from rasterio.io import MemoryFile
from urllib.error import HTTPError

GEOTIFF_MIME_TYPES = ['application/geotiff', 'image/tiff;application=geotiff', 'image/geo+tiff']


def _check_bands(collection, bands):
    """Check if bands are in rangetype"""

    if bands:
        band_names = [band.strip() for band in bands.split(",")]
        collection_rangetype_url = collection + '/coverage/rangetype?f=json'
        try:
            with requests.get(collection_rangetype_url) as request:
                collection_rangetype_json = request.json()
                fields = []
                for field in collection_rangetype_json['field']:
                    fields.append(field['name'])
                for band in band_names:
                    if band not in fields:
                        msg = "Band '{}' is not provided by the collection.".format(band)
                        raise Exception(msg)
        except HTTPError as err:
            msg = "Requesting collection rangetype at '{}' failed: '{}'".format(collection_rangetype_url, err)
            raise Exception(msg)


def _get_geotiff_format(collection):
    """Get format string for geotiff defined by the server from the links"""

    collection_url = collection + '?f=json'
    try:
        with requests.get(collection_url) as request:
            request.raise_for_status()
            collection_json = request.json()
            collection_links = collection_json.get('links', None)
            if collection_links:
                format_geotiff = None
                for link in collection_links:
                    if link['type'].replace(" ", "") in GEOTIFF_MIME_TYPES:
                        format_geotiff = link['href'].split('f=')[-1]
                if format_geotiff is None:
                    msg = "No link found for collection '{}' to get coverage data as geotiff.".format(collection)
                    raise Exception(msg)
                else:
                    return format_geotiff
            else:
                raise Exception("The collection '{}' has no links.".format(collection))
    except HTTPError as err:
        msg = "Requesting collection metadata for collection '{}' failed: {}".format(collection_url, err)
        raise Exception(msg)


def get_coverage(collection, bbox, bands=None):
    """Download coverage data as geotiff"""

    if collection.endswith('/'):
        collection = collection[:-1]
    format_geotiff = _get_geotiff_format(collection)
    _check_bands(collection, bands)

    # Request coverage data
    if bands:
        coverage_download_url = collection + '/coverage?f={}&bbox={}&range-subset={}'\
            .format(format_geotiff, ','.join(map(str, bbox)), bands)
    else:
        coverage_download_url = collection + '/coverage?f={}&bbox={}'\
            .format(format_geotiff, ','.join(map(str, bbox)))

    print("Requesting coverage from '{}'".format(coverage_download_url))
    try:
        with requests.get(coverage_download_url, verify=True, stream=True) as request:
            request.raise_for_status()

            return MemoryFile(request.content)
    except HTTPError as err:
        msg = "Requesting coverage data from '{}' failed: '{}'".format(coverage_download_url, err)
        raise Exception(msg)
