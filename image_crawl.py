import argparse, io, datetime, inspect, re, sys, os, uuid, tqdm
import ctypes, shutil, time
from collections import defaultdict, namedtuple
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import requests
from multiprocessing import Process, Value, Queue, Lock
from queue import Empty, Full
from PIL import Image
from josefutils import get_file_contents_v4, save_to_file, print_exception, glob_image_files

URLFields = namedtuple('URLFields', ['http_or_https', 'fields'])


class Configurations:
    """
    This is the configuration object which will be used almost everywhere.
    """

    def __init__(self):
        self.working_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        self.log_file = os.path.splitext(os.path.abspath(sys.argv[0]))[0] + '.log'
        self.supported_image_formats = [".jpg", ".png", ".jpeg"]
        self.website_root_dir = None  # self.website_root_dir = args.root_save + self.dir_from_url
        self.dir_from_url = None
        self.url = None  # Provided by args
        self.start_from = None
        self.visited_urls = set()  # urls already visited, provided by ini file and updated by program
        self.image_urls = set()  # image urls already downloaded, updated by program
        self.image_urls_history = set()  # image urls already downloaded previously, provided by ini
        self.images_downloaded = None  # Total number of images downloaded, updated by the download workers
        self.dir_marked = 'images.marked'
        self.dir_images = 'images'
        self.dir_large = 'images.large'
        self.dir_labels = 'labels'
        self.visited_urls_log = None  # A base name, NOT full path, set up in self.update_from
        self.downloaded_log = None  # A base name, NOT full path, set up in self.update_from
        self.download_processes: list[Process] = []
        self.locked_url = None
        self.allowed_urls = None
        self.censored_words = []  # Provided by file named censor.ini
        self.unwanted_starts = ['javascript', 'webcal:', 'file:', 'mailto:', 'tel:', 'ftp:', 'magnet:', '#', '?',
                                'whatsapp:', 'twitter:', 'instagram:', 'weixin:', 'weibo:', 'sms:']
        self.excepted_urls = []  # Provided by args, a file called excepts.ini or both
        self.connection_failures = 0
        self.max_failures = None  # Maximum failure torlerence, provided by args

    def update_from_cmd_args(self, args: argparse.Namespace):
        """
        Copy all the keys/values of args into Configurations
        """
        for key, value in vars(args).items():
            setattr(self, key, value)
        if args.excepts:
            [self.excepted_urls.append(item) for item in args.excepts]
        self.update_excepts_and_censor_ini()
        self.__setup_locked_url_or_start_from()
        self.dir_from_url = dir_name_from_url(url=self.url)
        self.visited_urls_log = f"crawl-visited-urls-{self.dir_from_url}.ini"
        self.downloaded_log = f"crawl-downloaded-{self.dir_from_url}.ini"
        four_dirs = [self.dir_marked, self.dir_images, self.dir_large, self.dir_labels]
        self.website_root_dir = os.path.join(self.root_save, self.dir_from_url)
        [os.makedirs(os.path.join(self.website_root_dir, d), exist_ok=True) for d in four_dirs]

    def update_excepts_and_censor_ini(self, printout=True):
        excepts_ini = os.path.join(self.working_path, "excepts.ini")
        if os.path.isfile(excepts_ini):
            print(f"\033[35m{excepts_ini}\033[0m found, adding contents to excepted_urls ...") if printout else ()
            excepts = get_file_contents_v4(excepts_ini, remove_comments=True)
            [self.excepted_urls.append(item) for item in excepts]
            self.excepted_urls = list(set(self.excepted_urls))
            self.excepted_urls.sort()
            save_to_file(file_name=excepts_ini, contents=self.excepted_urls)
        censor_ini = os.path.join(self.working_path, "censor.ini")
        if os.path.isfile(censor_ini):
            print(f"\033[35m{censor_ini}\033[0m found, adding keywords to censor from links ...") if printout else ()
            censored_words = get_file_contents_v4(censor_ini, remove_comments=True)
            [self.censored_words.append(item.lower()) for item in censored_words]
            self.censored_words = list(set(self.censored_words))
            self.censored_words.sort()
            save_to_file(file_name=censor_ini, contents=self.censored_words)

    def __setup_locked_url_or_start_from(self):
        # "https://www.cbc.ca/kidsnews/" ->kidsnews fields = ['www.cbc.ca', 'kidsnews']
        http_or_https, fields = fields_in_url(self.url)
        base_url = http_or_https + self.url.rstrip('/').replace(http_or_https, "").split('/')[0]
        self.url = base_url + '/'
        if len(fields) > 1:
            if self.no_lock:  # We interpret subdomains as start_from
                self.start_from = '/' + '/'.join(fields[1:])
            else:  # We interpret subdomains as locked_to subdomains
                self.locked_url = '/' + '/'.join(fields[1:])
                fields_without_html = [field for field in fields if not field.endswith(('html', 'htm', 'php', 'shtml'))]
                if '.php?' in fields_without_html[-1]:
                    fields_without_html = fields_without_html[:-1]
                locked_subdomain = '/' + '/'.join(fields_without_html[1:])
                self.allowed_urls = [locked_subdomain, urljoin(self.url, locked_subdomain),
                                     re.sub(r"https://|http://", "//", urljoin(self.url, locked_subdomain))]
                if self.custom_lock:
                    for item in self.custom_lock:
                        self.allowed_urls.append(item)
                        self.allowed_urls.append(urljoin(self.url, item))
                        self.allowed_urls.append(re.sub(r"https://|http://", "//", urljoin(self.url, item)))
                print(f"\033[1;31mSearch is restricted within \033[36m{self.allowed_urls}\033[0m")

    def debug_print(self, *argv, **kwargs):
        caller = inspect.currentframe().f_back.f_code.co_name
        print(f"🟡\033[35m{caller}:\033[0m🟡", *argv, **kwargs) if self.debug else ()

    def __repr__(self):
        string_list = [f"Save to = \033[36m{self.website_root_dir}\033[0m",
                       f"url = \033[36m{self.url}\033[0m",
                       f"Excepted urls = \033[31m{self.excepted_urls}\033[0m",
                       f"Censored words= \033[31m{self.censored_words}\033[0m"]
        if self.start_from:
            string_list.append(f"Start from = \033[1;36m{self.start_from}\033[0m")
        return '\n'.join(string_list)


# region ✅ First level Functions
def dir_name_from_url(*, url):
    """
    Replace characters in url to make it suitable for a directory name
    'https://www.reuters.com/'  -> 'www_reuters_com'
    :param url:
    :return:
    """
    dir_name = re.sub(r"http.*//(.*)", r"\1", url.rstrip('/'))
    dir_name = re.sub(r"[^a-zA-Z0-9]", r"_", dir_name)
    return dir_name


def fields_in_url(url):
    """
    https://kids.nationalgeographic.com/games/funny-fill-in/article/funny-fill-in-the-fast-and-the-flurryous
    ['kids.nationalgeographic.com', 'games', 'funny-fill-in', 'article', 'funny-fill-in-the-fast-and-the-flurryous']
    :param url:
    :return: the fields split by '/'
    """
    http_or_https = url.split('//')[0] + '//'
    fields = url.rstrip('/').replace(http_or_https, "").split('/')
    return URLFields(http_or_https, fields)


def keep_only_unique_urls(visited_urls: list[str]) -> list[str]:
    """
    Remove the first level path from URLs that end with 'html', 'htm', or 'shtml'.
    URLs that have the same first level path are considered categories, not webpages.
    :param visited_urls: List of visited URLs
    :return: List of URLs not categories, but ony webpages
    """
    if not visited_urls:
        return visited_urls
    visited_urls = list(visited_urls)
    http_or_https = visited_urls[0].split('//')[0] + '//'
    index_html_urls = [f"index{i}.html" for i in range(1, 10)]
    index_html_urls += [f"index{i}.htm" for i in range(1, 10)]
    index_html_urls += [f"index{i}.shtml" for i in range(1, 10)]
    index_html_urls += [f"index{i}.php" for i in range(1, 10)]
    urls_with_html = [url for url in visited_urls if url.endswith(('html', 'htm', 'shtml', 'php')) and
                      not any([url.endswith(k) for k in index_html_urls])]
    urls_no_html = [url.rstrip('/') + '/' for url in visited_urls if not url.endswith(('html', 'htm', 'shtml'))]
    url_counts = defaultdict(int)
    for url in urls_no_html:
        fields = url.rstrip('/').replace(http_or_https, "").split('/')
        for pos in range(1, len(fields)):
            sub_fields = fields[:pos + 1]
            key = http_or_https + "/".join(sub_fields) + '/'
            url_counts[key] += 1
    web_pages = [k for k, v in url_counts.items() if v == 1 and k in urls_no_html]
    return sorted(web_pages + urls_with_html, key=len)


def save_visited_url_and_downloaded_images(cfg: Configurations, del_empty=True, print_info=True):
    if cfg.dry_run:
        return
    if (del_empty and cfg.images_downloaded == 0
            and not os.path.isfile(os.path.join(cfg.working_path, cfg.downloaded_log))):
        shutil.rmtree(cfg.website_root_dir)
        print(f"\033[31m{cfg.website_root_dir}\033[0m was deleted ...")
        return
    visited_url_log = os.path.join(cfg.working_path, cfg.visited_urls_log)
    downloaded_log = os.path.join(cfg.working_path, cfg.downloaded_log)
    visited_urls = list(cfg.visited_urls)
    visited_urls = keep_only_unique_urls(visited_urls)
    downloaded_images = list(cfg.image_urls_history) + list(cfg.image_urls)
    save_to_file(visited_url_log, visited_urls, append=False)
    save_to_file(downloaded_log, downloaded_images, append=False)
    if print_info:
        b = os.path.basename
        print(f"\033[35m{b(visited_url_log)}\033[0m and \n\033[35m{b(downloaded_log)}\033[0m saved.")


def custom_sort_key(string: str) -> int:
    """
    Sort the strings in the desired order
    1) First, the strings ending with 'html', 'htm' or 'shtml'
    2) Second, the length of the string
    In this way, the html pages will be at the end of the list
    :param string: the column name of the dataframe
    :return: a string processed to place them in the desired order when sorting
    """
    return len(string) + 2 ** 32 if string.endswith(('html', 'htm', 'shtml')) else len(string) - 2 ** 32


# endregion


# region ✅ Secondary functions
def early_exit_by_ctrl_c(cfg: Configurations, show_hint=True):
    print("Ctrl +C was pressed ...") if show_hint else ()
    this_func = inspect.currentframe().f_code.co_name
    pids = [p.pid for p in cfg.download_processes if p is not None]
    [p.terminate for p in cfg.download_processes if p is not None]
    save_visited_url_and_downloaded_images(cfg)
    print(f"\033[35m{this_func}\033[0m terminated download process with \033[36m{pids}\033[0m as well.")
    sys.exit(201)


# endregion

# region ✅ Major functions
def is_link_acceptable(cfg, link_href: str):
    """
    Check if a link is acceptable based on certain criteria defined in the configuration object.
    Args:
        cfg (Configuration): The configuration object containing settings.
        link_href (str): The URL of the link to check.
    Returns:
        bool: True if the link is acceptable, False otherwise.
    """
    http_or_https = cfg.url.split('//')[0] + '//'
    base_url = http_or_https + cfg.url.rstrip('/').replace(http_or_https, "").split('/')[0]
    href = link_href.rstrip('/')
    if href.startswith('//'):
        href = urljoin(cfg.url, href)
    url_without_base = href.replace(base_url, "")
    # Check if the URL is excluded
    if cfg.locked_url:
        if not any([link_href.startswith(k) for k in cfg.allowed_urls]):
            return False
    # Check if the URL starts with unwanted patterns
    elif any([link_href.strip().lower().startswith(k) for k in cfg.excepted_urls + cfg.unwanted_starts]):
        return False
    elif any([url_without_base.strip().startswith(k) for k in cfg.excepted_urls + cfg.unwanted_starts]):
        return False
    elif any([k in link_href.lower() for k in cfg.censored_words]):
        return False
    elif 'http' in url_without_base:
        return False
    else:
        last_field = href.split('/')[-1]
        if '#' in last_field:
            return False
        m_dot_something = re.search(r"\.([^.]+)$", last_field)
        if m_dot_something:
            if m_dot_something[1] not in ["html", "htm", "shtml"]:
                return False
    return True


def is_image_already_downloaded(dir_images, img_basename):
    # f"%Y%m%d-%H%M%S-{uuid_style_name}"  20230911-000000-, totally 16 characters
    all_images = glob_image_files(dir_images)
    all_images = [os.path.basename(f)[16:] for f in all_images]
    if img_basename[16:] in all_images:
        return True
    else:
        return False


def download_image(dir_images, dir_large, img_url, supported_image_formats,
                   min_image_width, min_image_height, use_uuid_style=False):
    """
    Download an image from a given URL and save it to the specified directories.
    Args:
        dir_images (str): The directory to save the image.
        dir_large (str): The directory to save the large version of the image.
        img_url (str): The URL of the image to download.
        supported_image_formats (list): List of supported image formats.
        :param use_uuid_style:
    Returns:
        bool: True if the image is downloaded and saved successfully, False otherwise.
    """
    # Check if the image URL is provided
    if not img_url:
        return False
    # Extract the image name from the URL
    img_name = img_url.split('?')[0].split('/')[-1]
    # Extract the base name and extension of the image
    base, extension = os.path.splitext(os.path.basename(img_name))
    # Check if the image extension is supported
    if extension.lower() not in supported_image_formats:
        return False
    if use_uuid_style:
        uuid_str = str(uuid.uuid4())[:32]
        base = base[:13]
        uuid_style_name = base.lower() + uuid_str[len(base):]
        uuid_style_name = re.sub(r"[^A-Za-z0-9]", "-", uuid_style_name)
        uuid_style_name = uuid_style_name.lower() + extension.lower()
    else:
        base = base[:64]  # Filename too long may cause error
        uuid_style_name = re.sub(r"[^A-Za-z0-9]", "-", base).lower() + extension.lower()
    # Combine the truncated base name, unique identifier, and extension to create the image basename
    img_basename = datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S-{uuid_style_name}")
    if is_image_already_downloaded(dir_images, img_basename):
        print(f"\033[31m{img_basename}\033[0m is already downloaded.")
        return False
    # Create the full path for the image and the large version of the image
    img_full_path = os.path.join(dir_images, img_basename)
    img_large_full_path = os.path.join(dir_large, img_basename)
    try:
        # Send a request to download the image
        response = requests.get(img_url, timeout=15)
        # Check if the request was successful
        if response.status_code == 200:
            # Get the content type of the response
            content_type = response.headers.get('Content-Type', '')
            # Check if the response contains an image
            if content_type.startswith('image/'):
                # Get the image data
                image_data = response.content
                try:
                    # Try to open and convert the image data
                    img_np = Image.open(io.BytesIO(image_data)).convert('RGB')
                    # Get the width and height of the image
                    width, height = img_np.size
                    # Check if the image is large enough
                    if width >= min_image_width and height >= min_image_height:
                        # Save the image to the large directory
                        with open(img_large_full_path, "wb+") as file:
                            file.write(image_data)
                        # Resize the image
                        img_np.thumbnail((min_image_width, min_image_height), Image.Resampling.LANCZOS)
                        # Save the image to the directory
                        img_np.save(img_full_path)
                        return True
                except Exception as e:
                    print_exception(e)
                    return False
        return False
    except Exception as e:
        print_exception(e)
        return False


def download_image_worker(*args):
    """
    This function runs as a separate PROCESS
    Args:
        *args: Variable-length argument list. Expected arguments in order:
            - dir_images (str): The directory to save the downloaded images.
            - dir_large (str): The directory for large images.
            - images_downloaded (Value): A shared value to track the number of downloaded images.
            - connection (Connection): A connection object to receive image URLs.
            - dry_run (bool): Flag indicating if it's a dry run.
            - supported_image_formats (list): List of supported image formats.
            - min_image_width, min_image_height (int): Minimum width and height of the image.
    Returns:
        None
    """
    dir_images, dir_large = args[0], args[1]
    images_downloaded: Value = args[2]
    connection: Queue = args[3]
    lock: Lock = args[4]
    dry_run: bool = args[5]
    supported_image_formats = args[6]
    min_image_width, min_image_height = args[7], args[8]
    if dry_run:  # If it's a dry run, print message and return
        print("\033[1;31mDry running, downloading finished...\033[0m")
        return
    while True:
        try:
            image_url = connection.get(block=True, timeout=1)
        except Empty:
            time.sleep(1)
            continue
        except Exception as e:
            print_exception(e)
            continue
        try:
            if download_image(dir_images, dir_large, image_url, supported_image_formats,
                              min_image_width, min_image_height, use_uuid_style=False):
                with lock:
                    images_downloaded.value += 1
        except Exception as e:
            print_exception(e)
            continue
        if image_url is None:
            # Signal to break the loop when all images have been processed
            break


def crawl_website(*, cfg: Configurations, depth: int, connection: Queue, images_downloaded, url):
    visited_urls = cfg.visited_urls
    image_urls = cfg.image_urls
    stack = [(url, depth)]
    counter = 0
    while stack:
        counter += 1
        url, depth = stack.pop() if cfg.depth_first else stack.pop(0)
        visited_urls.add(url)  # Update visited_urls here
        cfg.images_downloaded = images_downloaded.value
        try:
            try:
                response = requests.get(url, timeout=30)
                response.encoding = 'utf-8'
                soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
            except Exception as e:
                print_exception(f"{str(e)}@\033[36m{url}\033[0m")
                continue
            img_tags = soup.find_all('img')
            for img_tag in img_tags:
                img_src = img_tag.get('src')
                if img_src:
                    img_url = urljoin(url, img_src)
                    if img_url not in cfg.image_urls_history and img_url not in image_urls:
                        image_urls.add(img_url)
                        try:
                            connection.put(img_url) if not cfg.dry_run else ()
                        except Full:
                            print(f"\033[1;31mQueue is full, waiting 60 seconds...\033[0m")
                            time.sleep(60)
                        except Exception as e:
                            print_exception(e)
            if sys.platform.lower() == 'darwin':
                qsize = -1  # macOS doesn't have a queue size
            else:
                qsize = connection.qsize()
            time.sleep(qsize * 0.1) if qsize > 10 else ()
            link_tags = soup.find_all('a', href=True)
            link_tags = [link for link in link_tags if
                         link['href'] and link['href'] and is_link_acceptable(cfg, link['href'])]
            link_tags = [e for idx, e in enumerate(link_tags) if
                         e['href'] not in [ee['href'] for ee in link_tags][:idx]]
            link_tags.sort(key=lambda link_tag_: custom_sort_key(link_tag_['href']),
                           reverse=False if cfg.depth_first else True)
            current_depth_urls = [u for u, d in stack if d == depth]
            len_current_depth = len(current_depth_urls)
            link_tags = [e for e in link_tags if urljoin(url, e['href']) not in visited_urls]
            if len(link_tags):
                print("")
            fmt_string = (f"\033[36m{len_current_depth}/\033[1;36m{len(stack)}, Q: {qsize:2d}\033[0m "
                          f"E: \033[1;31m{cfg.connection_failures}\033[0m "  # Connection errors
                          f"I: \033[35m{len(image_urls)} \033[0m"  # Images found
                          f"D: \033[36m{images_downloaded.value}\033[0m")  # Images downloaded
            now = f"\033[35m{datetime.datetime.now().strftime(f'%Y%m%d-%H:%M:%S')}\033[0m"
            digits_of_depth = max(len(str(depth)), len(str(depth + 1)))
            info_str = (f"{now}(\033[36m{counter}\033[0m)Crawling:\033[32m{url[:60]:60}\033[0m\033[1;32m"
                        f"({depth:<{digits_of_depth}d}) \033[0m{fmt_string}")
            print(info_str)
            for link_tag in link_tags:
                link_href = link_tag['href']
                now = f"\033[35m{datetime.datetime.now().strftime(f'%Y%m%d-%H:%M:%S')}\033[0m"
                link_output = link_href.replace(cfg.url.rstrip('/'), '')
                print(f"{now}(\033[36m{counter}\033[0m)Adding  :{link_output[:60]:60}\033[1;36m"
                      f"({depth + 1:<{digits_of_depth}d}) \033[0m{fmt_string}")
                url_to_append = urljoin(url, link_href)
                stack.append((url_to_append, depth + 1)) if url_to_append.startswith(cfg.url) else ()
            if counter % 200 == 0:
                now = f"\033[35m{datetime.datetime.now().strftime(f'%Y%m%d-%H:%M:%S')}\033[0m"
                print(f"\n{now}(\033[36m{counter}\033[0m) {fmt_string} \033[1;35mSaving/loading ini files...\033[0m")
                save_to_file(cfg.log_file, [info_str], append=True)
                save_visited_url_and_downloaded_images(cfg, del_empty=False, print_info=False)
                cfg.update_excepts_and_censor_ini(printout=False)  # Reload the excepts and censored words
        #except requests.exceptions.RequestException as e:
        except Exception as e:
            print_exception(e)
            time.sleep(cfg.connection_failures * 60)
            cfg.connection_failures += 1
            if cfg.connection_failures >= cfg.max_failures:
                print(f"Reached maximum connection failures ({cfg.max_failures}). Exiting.")
                early_exit_by_ctrl_c(cfg, show_hint=False)


def operations_worker(cfg: Configurations):
    """
    Function to perform operations on a website.

    Args:
        cfg (Configurations): An instance of the Configurations class.

    Returns:
        None
    """
    # File paths
    visited_url_log = os.path.join(cfg.working_path, cfg.visited_urls_log)
    downloaded_log = os.path.join(cfg.working_path, cfg.downloaded_log)

    # Load visited URLs
    cfg.visited_urls = set(get_file_contents_v4(visited_url_log))

    # Remove URLs with one level depth
    new_visited_urls = keep_only_unique_urls(cfg.visited_urls)
    cfg.visited_urls = set(new_visited_urls)

    # Load image URLs history
    cfg.image_urls_history = set(get_file_contents_v4(downloaded_log))

    # Remove the old log file
    if not cfg.keep_log:
        os.remove(cfg.log_file) if os.path.exists(cfg.log_file) else ()

    queue = Queue()
    lock = Lock()
    # File directories
    dir_images = os.path.join(cfg.website_root_dir, cfg.dir_images)
    dir_large = os.path.join(cfg.website_root_dir, cfg.dir_large)

    # Initialize images downloaded counter
    images_downloaded = Value(ctypes.c_int64, 0)

    # Supported image formats
    supported_image_formats = cfg.supported_image_formats

    # Arguments for the download subprocess
    args = (dir_images, dir_large, images_downloaded, queue, lock, cfg.dry_run, supported_image_formats,
            cfg.min_image_width, cfg.min_image_height)

    # Create and start the download subprocess
    num_workers = max(min(cfg.num_workers, 20), 1)
    download_sub_processes = [Process(target=download_image_worker, args=args) for _ in range(num_workers)]
    cfg.download_processes = download_sub_processes  # Save for termination with Ctrl+C
    [p.start() for p in download_sub_processes]

    # Determine the URL to crawl
    if cfg.locked_url:
        url = urljoin(cfg.url, cfg.locked_url)
    elif cfg.start_from:
        url = urljoin(cfg.url, cfg.start_from)
    else:
        url = cfg.url
    if not cfg.skip_alert:
        print(f"\033[36mCrawling will start in 30 seconds ...\033[0m")
        time.sleep(2)
        for _ in tqdm.trange(30):  # Wait some while to view to configurations
            time.sleep(1)

    # Crawl the website
    crawl_website(cfg=cfg, depth=0, connection=queue, images_downloaded=images_downloaded, url=url)

    # Signal the end of crawling
    [queue.put(None) for _ in range(num_workers + 1)]

    # Wait for the download subprocess to complete
    [p.join() for p in download_sub_processes]

    # Print the total number of downloaded images
    print(f"\033[35mImages downloading finished with totally "
          f"\033[0m{images_downloaded.value}\033[35m images downloaded ...\033[0m")

    cfg.images_downloaded = images_downloaded.value
    # Save the visited URLs and downloaded images
    save_visited_url_and_downloaded_images(cfg)


# endregion


# region ✅ Program Entry
def parse_arguments():
    description = ("This program will crawl website and download images at desired size in separate processes. \n"
                   "\033[31m1\033[0m. It will save visited URLs and downloaded images into 2 ini files. Make sure "
                   "you delete them if you want to crawl the whole website again ignoring links visited before. \n"
                   "\033[31m2\033[0m. A file called excepts.ini with all unwanted urls can prevent crawling into those "
                   "links. Each line is an excepted link in format like \033[35m/football/\033[0m. A sample ini can be "
                   "generated with the command \033[35m--excepts_sample\033[0m. You may edit the excepts.ini "
                   "in the process of crawling, the program will reload it every 200 rounds, the same case with "
                   "censor.ini. \n"
                   "\033[31m3\033[0m. A file called censor.ini with all unwanted keywords in the link can prevent "
                   "crawling into those links. Unlike excepts.ini, censor.ini search the whole link, not only the"
                   " start. \n"
                   "\033[31m4\033[0m. A log file will record the progress of crawling. \n"
                   "\033[31m5\033[0m. Normally if your url has certain subdomain of a website, the crawler will "
                   "lock into this subdomain. Eg. www.some_website.com/football, the crawler will not crawl the whole "
                   "website, but only the subdomain /football. To cancel this, use --no_lock. Or you may add "
                   "custom locks with --custom_lock, to add more subdomains to crawl. You may add multiple "
                   "subdomains. eg. --custom_lock /tennis/ /basketball/ /golf/ \n"
                   "\033[31m6\033[0m. Please be noted that queue size does not work in Mac OS. It will always show "
                   " -1. \n"
                   "\033[31m7\033[0m. Please be noted that some websites use Javascript to generate pages, and we "
                   "don't interpret Javascript. So those websites will not be crawled. \n"
                   "\033[31m8\033[0m. Many websites may enforce anit-crawling acts, we did not take any measures "
                   "to prevent this. \n"
                   "\033[31m9\033[0m.By default, crawling will work in width first manner, which means it will "
                   "finish crawling current page first before diving deeper, you may change it by --depth_first.\n"
                   "\033[31m10\033[0m. You may watch the crawling process and add some excepts to excepts.ini by "
                   "echo -e '/some_sub_domain/' | tee -a /path_of_this_program/excepts.ini. It will take effect in "
                   "next 200 rounds. \033[1;31mThis is very useful to save time.\033[0m\n"
                   "\033[31m11\033[0m.  (4     ) 312/2025, Q: 47 E: 0 I: 815 D: 324, "
                   "4 is depth, 312 is current depth urls left in stack, 2025 is total stack to crawl, Q: is length of "
                   "image urls still in the queue to be downloaded by download workers. E: is connection errors, "
                   "I: is total image links found, D: is images already downloaded")
    helps = {'excepts': ("A list of links not to crawl, for convenience, you can edit "
                         "a file named \033[35mexcepts.ini\033[0m and put all excepts into it. "
                         "Each line an excepted link"),
             'dry_run': ("Dry run, iterate over links only, no downloading. "
                         "\033[1;31mIt is highly recommended\033[0m to run this first and put unwanted "
                         "links into \033[35mexcepts.ini\033[0m"),
             'excepts_sample': "Generate a sample \033[35mexcepts.ini\033[0m in current dir",
             'num_workers': ("The number of download image workers(1-20). Watch the output for the queue size as "
                             "\033[36m3555/36433,Q: 10\033[0m, and try to add workers to keep the queue under 10"),
             'depth_first': ("Crawl in depth first manner (\033[31mnot recommended\033[0m), because it may dive into "
                             "very old pages first linked by other pages.")
             }
    parser = argparse.ArgumentParser(description=description, prog='apva',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-V', '--version', action='version', version='%(prog)s 1.01, Build 2023 Vintage')
    parser.add_argument("-D", "--debug", help="Print extra debug information", action="store_true")
    parser.add_argument("url", help="\033[1;31murl\033[0m to craw for pictures", type=str)
    parser.add_argument("--dry_run", help=helps['dry_run'], action="store_true")
    # Other args from this line, lines before should be kept
    parser.add_argument("--root_save", default='/media/usb0/crawls/', help="Root dir to save", type=str)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-S", "--start_from", help="The url we start from instead of the homepage", type=str)
    group.add_argument("-N", "--no_lock", help="Do not lock url to subfields even url has subdomains",
                       action="store_true")
    group.add_argument("-C", "--custom_lock", nargs='+', help="Except locked url, more fields are allowed", type=str)
    parser.add_argument("-W", "--num_workers", metavar='N', help=helps['num_workers'], default=3, type=int)
    parser.add_argument("-M", "--max_failures", metavar='N', help="Try N times if connection fails",
                        type=int, default=10000)
    parser.add_argument("--min_image_width", metavar='N', help="Minimum image width to download",
                        default=640, type=int)
    parser.add_argument("--min_image_height", metavar='N', help="Minimum image height to download",
                        default=640, type=int)
    parser.add_argument("-K", "--keep_log", help="Keep existing log file", action="store_true")
    parser.add_argument("-SA", "--skip_alert", help="Skip 30 seconds alert", action="store_true")
    parser.add_argument("-DF", "--depth_first", help=helps['depth_first'], action="store_true")
    parser.add_argument("--excepts", nargs='+', help=helps['excepts'], type=str)
    parser.add_argument("--excepts_sample", help=helps['excepts_sample'], action="store_true")
    return parser.parse_args()


def main(cfg: Configurations):
    """
    This is the main function that performs the desired operations based on the given configurations.
    Args:
        cfg (Configurations): The configurations object containing the program settings.
    Returns:
        None
    """
    args = parse_arguments()
    if args.excepts_sample:  # Generate a sample excepts.ini file and print its content
        print("A sample excepts.ini was generated in current dir.")
        lines = ["/register", "/sitemap", "/about", "/signup", "/login", "/account"]
        [print(line) for line in lines]
        save_to_file("excepts.ini", lines, append=False)
        return
    if args.url is None:  # Check if the URL argument is provided
        print("\033[31m--url MUST be provided\033[0m")
        return
    cfg.update_from_cmd_args(args)  # Update important import of configurations from args
    print(f"\033[32mConfigurations are:\033[0m\n{cfg}")
    # Perform the desired operations based on the configurations
    operations_worker(cfg)


if __name__ == "__main__":
    configurations = Configurations()
    try:
        main(configurations)
    except KeyboardInterrupt:
        early_exit_by_ctrl_c(configurations)
# endregion

