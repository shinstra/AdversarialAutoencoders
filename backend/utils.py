
__all__ = ['generate_cache_folder']

import os
from datetime import datetime


def generate_cache_folder(path, tag=''):
    """
    Generates a timestamped folder.
    :param path: path to create folder in.
    :param tag: Additional tag to be appended to beginning of folder name.
    :return: path to the newly created folder.
    """

    assert os.path.isdir(path), 'Cannot create folder in non-existent location.'

    time_str = datetime.strftime(datetime.today(), "%Y-%m-%d-%H-%M-%S-%f")
    folder_name = tag + '_' + time_str
    folder_name.replace(' ', '_')  # Change spaces to underscores
    folder_path = os.path.join(path, folder_name)

    assert os.path.exists(folder_path) is False, 'How can you spam the run button so fast? it is timestamped to the ms!'

    os.mkdir(folder_path)
    return folder_path