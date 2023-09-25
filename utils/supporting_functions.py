import re, inspect, glob


def get_file_contents_v4(file_name: str, remove_comments: bool = False, ansi_only: bool = False) -> [str]:
    """
    :param file_name: The file to read as text file
    :param remove_comments: comments with '#' will be removed if set
    :param ansi_only: remove non ansi characters which are most likely colour codes
    :return: a list of str
    """
    try:
        ansi_escapes = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        with open(file_name, 'r') as f:
            lines = [line.strip() for line in f]  # remove left and right white spaces and '\n'
            lines = [re.sub(ansi_escapes, '', line).strip() for line in lines] if ansi_only else lines
            lines = [re.sub("#.*", '', line).strip() for line in lines] if remove_comments else lines
            lines = [line for line in lines if line]  # exclude empty lines
            return lines
    except Exception as e:
        print(f"\033[1;31mException: {str(e)}\033[0m")
        return []


def save_to_file(file_name: str, contents: list[str] = [], append=False):
    f = open(file_name, 'a') if append else open(file_name, 'w')
    [f.write(line + '\n') for line in contents]
    f.close()


def print_exception(e: Exception):
    caller = inspect.currentframe().f_back.f_code.co_name
    print(f"❌\033[1;31mException from \033[35m{caller}\033[1;31m: {str(e)}\033[0m❌")


def glob_image_files(source: str, formats=['jpg', 'png', 'jpeg'], filters=None, reverse: bool = False):
    """
    Search <source> for image files recursively, both lower case and upper case files
    will be included in result
    :param filters: A fileter string to be used at the front of the file name
    :param source:
    :param formats:
    :param reverse, whether to return images in reversed order
    :return: a sorted list of images
    """
    image_files = []
    if filters:
        for format_ in formats:
            image_files_of_format_1 = glob.glob(source.rstrip('/') + f"/{filters}*.{format_.lower()}")
            image_files_of_format_2 = glob.glob(source.rstrip('/') + f"/{filters}*.{format_.upper()}")
            image_files = image_files + image_files_of_format_1 + image_files_of_format_2
    else:
        for format_ in formats:
            image_files_of_format_1 = glob.glob(source.rstrip('/') + f"/*.{format_.lower()}")
            image_files_of_format_2 = glob.glob(source.rstrip('/') + f"/*.{format_.upper()}")
            image_files = image_files + image_files_of_format_1 + image_files_of_format_2
    return sorted(image_files, reverse=reverse)
