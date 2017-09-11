import argparse
import logging

from unipath import Path
import coloredlogs

from argparseActions import InputDirectoryAction, OutputDirectoryAction

_default_input_path = Path('../paper')
args = None

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--image-directory',
                        action=InputDirectoryAction,
                        default=_default_input_path,
                        help='''The folder to recursively search for images to be converted.''')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help="increase output verbosity")
    parser.add_argument('-r', '--replace-existing',
                        action='store_true',
                        default=False,
                        help="Overwrite existing png files.")
    parser.add_argument('-e','--files-to-skip',
                        nargs='*',
                        help='PDF files that should not be converted',
                        default=list(),
                        required=False)
    return parser


def find_pdf_files(directory):
    def is_pdf_file(file):
        return file.ext == Path(u'.pdf')

    def is_not_in_exception_list(file):
        return not(file in args.files_to_skip)

    file_filter = lambda x : is_pdf_file(x) and is_not_in_exception_list(x)

    for file in directory.walk(filter=file_filter):
        print(file)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    coloredlogs.install()
    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.ERROR)
    logging.info('Running in verbose mode')

    find_pdf_files(args.image_directory)
